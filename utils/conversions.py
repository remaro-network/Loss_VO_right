import enum
import torch
from typing import Tuple
import torch.nn.functional as F
from utils import acos_linear_extrapolation

class QuaternionCoeffOrder(enum.Enum):
    XYZW = 'xyzw'
    WXYZ = 'wxyz'
    
def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    return _so3_exp_map(log_rot, eps=eps)[0]

def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square



def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    r"""Normalize a quaternion.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Args:
        quaternion: a tensor containing a quaternion to be normalized.
          The tensor can be of shape :math:`(*, 4)`.
        eps: small value to avoid division by zero.

    Return:
        the normalized quaternion of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = torch.tensor((1., 0., 1., 0.))
        >>> normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)



# based on:
# https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h

def hat_inv(h: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse Hat operator [1] of a batch of 3x3 matrices.

    Args:
        h: Batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`.

    Returns:
        Batch of 3d vectors of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `h` is of incorrect shape.
        ValueError if `h` not skew-symmetric.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = torch.abs(h + h.permute(0, 2, 1)).max()

    HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices is not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v

def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R T ]
        [ 0 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ hat(log_rotation) 0 ]
                         [   log_translation 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].

    Note that for any `log_transform` with `0 <= ||log_rotation|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_log_map(se3_exponential_map(log_transform)) == log_transform
    ```

    The conversion has a singularity around `||log(transform)|| = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid unstable gradients in the singular case.

    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.

    Raises:
        ValueError if `log_transform` is of incorrect shape.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., :3]
    log_rotation = log_transform[..., 3:]

    # rotation is an exponential map of log_rotation
    (
        R,
        rotation_angles,
        log_rotation_hat,
        log_rotation_hat_square,
    ) = _so3_exp_map(log_rotation, eps=eps)

    # translation is V @ T
    V = _se3_V_matrix(
        log_rotation,
        log_rotation_hat,
        log_rotation_hat_square,
        rotation_angles,
        eps=eps,
    )
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, 3, :3] = T
    transform[:, 3, 3] = 1.0

    return transform.permute(0, 2, 1)

def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[
                :, None, None
            ]
        )
    )

    return V

def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)

def so3_log_map(
    R: torch.Tensor, eps: float = 0.0001, cos_bound: float = 1e-4
) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.

    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: A float constant handling the conversion singularity.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call when computing `so3_rotation_angle`.
            Note that the non-finite outputs/gradients are returned when
            the rotation angle is close to 0 or π.

    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    phi = so3_rotation_angle(R, cos_bound=cos_bound, eps=eps)

    phi_sin = torch.sin(phi)

    # We want to avoid a tiny denominator of phi_factor = phi / (2.0 * phi_sin).
    # Hence, for phi_sin.abs() <= 0.5 * eps, we approximate phi_factor with
    # 2nd order Taylor expansion: phi_factor = 0.5 + (1.0 / 12) * phi**2
    phi_factor = torch.empty_like(phi)
    ok_denom = phi_sin.abs() > (0.5 * eps)
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    phi_factor[~ok_denom] = 0.5 + (phi[~ok_denom] ** 2) * (1.0 / 12)
    phi_factor[ok_denom] = phi[ok_denom] / (2.0 * phi_sin[ok_denom])

    log_rot_hat = phi_factor[:, None, None] * (R - R.permute(0, 2, 1))

    log_rot = hat_inv(log_rot_hat)

    return log_rot

def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4
    
def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to Rodrigues vector in radians.

    Args:
        rotation_matrix: rotation matrix of shape :math:`(N, 3, 3)`.

    Returns:
        Rodrigues vector transformation of shape :math:`(N, 3)`.

    Example:
        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_angle_axis(input)
        tensor([0., 0., 0.])

        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 0., -1.],
        ...                       [0., 1., 0.]])
        >>> rotation_matrix_to_angle_axis(input)
        tensor([1.5708, 0.0000, 0.0000])
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")
    quaternion: torch.Tensor = rotation_matrix_to_quaternion(rotation_matrix, order=QuaternionCoeffOrder.WXYZ)
    return quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)

def quaternion_to_angle_axis(
    quaternion: torch.Tensor, order: QuaternionCoeffOrder = QuaternionCoeffOrder.XYZW
) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation in radians.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion: tensor with quaternions.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.tensor((1., 0., 0., 0.))
        >>> quaternion_to_angle_axis(quaternion)
        tensor([3.1416, 0.0000, 0.0000])
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of {QuaternionCoeffOrder.__members__.keys()}")

    # unpack input and compute conversion
    q1: torch.Tensor = torch.tensor([])
    q2: torch.Tensor = torch.tensor([])
    q3: torch.Tensor = torch.tensor([])
    cos_theta: torch.Tensor = torch.tensor([])

    if order == QuaternionCoeffOrder.XYZW:
        q1 = quaternion[..., 0]
        q2 = quaternion[..., 1]
        q3 = quaternion[..., 2]
        cos_theta = quaternion[..., 3]
    else:
        cos_theta = quaternion[..., 0]
        q1 = quaternion[..., 1]
        q2 = quaternion[..., 2]
        q3 = quaternion[..., 3]

    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def quaternion_to_rotation_matrix(
    quaternion: torch.Tensor, order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ
) -> torch.Tensor:
    r"""Convert a quaternion to a rotation matrix.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 4)`.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        the rotation matrix of shape :math:`(*, 3, 3)`.

    Example:
        >>> quaternion = torch.tensor((0., 0., 0., 1.))
        >>> quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of {QuaternionCoeffOrder.__members__.keys()}")

    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    if order == QuaternionCoeffOrder.XYZW:
        x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)
    else:
        w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.0)

    matrix: torch.Tensor = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix



def rotation_matrix_to_quaternion(
    rotation_matrix: torch.Tensor, eps: float = 1.0e-8, order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ
) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) or (x, y, z, w) format.

    .. note::
        The (x, y, z, w) order is going to be deprecated in favor of efficiency.

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(*, 3, 3)`.
        eps: small value to avoid zero division.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        the rotation in quaternion with shape :math:`(*, 4)`.

    Example:
        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps,
        ...                               order=QuaternionCoeffOrder.WXYZ)
        tensor([1., 0., 0., 0.])
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of {QuaternionCoeffOrder.__members__.keys()}")


    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / torch.clamp(denominator, min=eps)

    # rotation_matrix_vec: torch.Tensor = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)
    rotation_matrix_vec: torch.Tensor = torch.reshape(rotation_matrix,(*rotation_matrix.shape[:-2], 9))

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0 + eps) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])