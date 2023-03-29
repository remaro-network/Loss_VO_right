import unittest
import torch
import os
from model.metric_functions.vo_metrics import SO3_chordal_metric, vector_distance, SE3_chordal_metric, quaternion_distance_metric
from model.loss_functions.pose_losses import se3_chordal_loss, mse_euler_pose_loss, quaternion_pose_loss
from utils.conversions import so3_exp_map, se3_exp_map, rotation_matrix_to_quaternion

class TestDatabaseDataloader(unittest.TestCase):
    def test_quaternion_pose_loss(self):
        ''' unit testing for pose loss w. orientation as quaternion'''
        # create a dummy data
        p1 = torch.zeros([1, 3, 7], device=torch.device('cuda:0'))
        p1[:,:,4] = 1 # 180 deg. at Z
        p2 = torch.eye(4, 4).unsqueeze(0).repeat(3, 1, 1).to(torch.device('cuda:0'))
        p2 = torch.reshape(p2, (3, 1, 4, 4))

        # call the function with data_dict
        loss_dict = quaternion_pose_loss({"result": p1, "poses": p2})

        # check the output
        self.assertEqual(first = loss_dict["loss"].shape, second = torch.Size([1]))
        self.assertAlmostEqual(loss_dict["loss"].item(), 1.41, delta = 0.01)
        self.assertEqual(first = loss_dict["rotation_loss"].shape, second = torch.Size([1]))
        self.assertAlmostEqual(loss_dict["rotation_loss"].item(), 1.41, delta = 0.01)
        self.assertEqual(first = loss_dict["traslation_loss"].shape, second = torch.Size([1]))
        self.assertAlmostEqual(loss_dict["traslation_loss"].item(), 0, delta = 0.01)
    
    def test_quaternion_distance_metric(self):
        ''' unit testing for quaternion distance function'''
        # create a dummy data
        q1 = torch.zeros(1, 4)
        q1[:, 3] = 1 # 180 deg. at Z
        q2 = torch.eye(4, 4).unsqueeze(0).repeat(3, 1, 1)
        q2 = torch.reshape(q2, (3, 1, 4, 4))
        distance = 0

        # run the function
        for i in range (q2.shape[0]):
            q_target = q2[i]
            q_target = rotation_matrix_to_quaternion(q_target[:,:3,:3])

            q_estimate = q1
            distance += quaternion_distance_metric(q_estimate, q_target)
        distance/=q2.shape[0]
        
        # check the output
        self.assertEqual(first = distance.shape, second = torch.Size([1]))
        self.assertAlmostEqual(distance.item(), 1.41, delta = 0.01)

    # @unittest.skip("skip test")
    def test_se3_chordal_loss(self):
        ''' unit testing for se3 chordal loss function'''
        # create a dummy data
        p1 = torch.zeros([1, 3, 6], device=torch.device('cuda:0'))
        p1[:, :, :3] = 1/torch.sqrt(torch.tensor(3.0))
        p1[:,:,-1] = torch.pi
        p2 = torch.eye(4, 4).unsqueeze(0).repeat(3, 1, 1).to(torch.device('cuda:0'))
        p2 = torch.reshape(p2, (3, 1, 4, 4))
        loss_dict = se3_chordal_loss({"result": p1, "poses": p2})
        
        # check the output
        self.assertEqual(first = loss_dict["loss"].shape, second = torch.Size([1]))
        self.assertAlmostEqual(loss_dict["loss"].item(), 8.77, delta = 0.1)
        self.assertEqual(first = loss_dict["rotation_loss"].shape, second = torch.Size([1]))
        self.assertAlmostEqual(loss_dict["rotation_loss"].item(), 8., delta = 0.1)
        self.assertEqual(first = loss_dict["traslation_loss"].shape, second = torch.Size([1]))
        self.assertAlmostEqual(loss_dict["traslation_loss"].item(), 0.77, delta = 0.01)

    # @unittest.skip("skip test")
    def test_SE3_chordal_metric(self):
        ''' unit testing for SE3 chordal distance function'''
        # create a dummy data
        p1 = torch.zeros([1, 3, 6],device=torch.device('cuda:0'))
        p1[:, :, :3] = 1/torch.sqrt(torch.tensor(3.0))
        p1[:,:,-1] = torch.pi
        p2 = torch.eye(4, 4).unsqueeze(0).repeat(3, 1, 1).to(torch.device('cuda:0'))
        p2 = torch.reshape(p2, (3, 1, 4, 4))
        distance = 0

        # run the function
        for i in range (p2.shape[0]):
            p_target = p2[i]
            p_estimate = se3_exp_map(p1[:, i, :])
            distance += SE3_chordal_metric(p_estimate, p_target)
        distance/=p2.shape[0]
        
        # check the output
        self.assertEqual(first = distance.shape, second = torch.Size([1]))
        self.assertAlmostEqual(distance.item(), 8.77, delta = 0.1)

    # @unittest.skip("skip test")
    def test_SO3_chordal_metric(self):
        ''' unit testing for SO3 chordal distance function'''
        # create a dummy data
        r1 = torch.zeros([1, 3, 6], device = torch.device('cuda:0'))
        r1[:, :, 2] = torch.pi
        r2 = torch.eye(4, 4).unsqueeze(0).repeat(3, 1, 1).to(torch.device('cuda:0'))
        r2 = torch.reshape(r2, (3, 1, 4, 4))
        distance = 0

        # run the function
        for i in range (r2.shape[0]):
            r_target = r2[i][:, :3, :3]
            r_estimate = so3_exp_map(r1[:, i, :3])
            distance += SO3_chordal_metric(r_estimate, r_target)
        distance/=r2.shape[0]

        # check the output
        self.assertEqual(first = distance.shape, second = torch.Size([1]))
        self.assertAlmostEqual(distance.item(), 8., delta = 0.01)

    # @unittest.skip("skip test")
    def test_vector_norm_metric(self):
        ''' unit testing for vector norm function'''
        # create a dummy data
        pretty_result = 1/torch.sqrt(torch.tensor(3.0))
        estimate = torch.ones(1, 3, 6)*pretty_result
        target = torch.zeros(3, 1, 4, 4)
        norm = 0.

        # run the function
        for i in range(estimate.shape[1]):
            norm += vector_distance(estimate[:, i, 3:], target[i][:,:-1,-1])
        norm/=estimate.shape[1]

        # check the output
        self.assertEqual(first = norm.shape, second = torch.Size([1]))
        self.assertAlmostEqual(norm.item(), 1.0, delta = 0.01)


if __name__ == '__main__':
    unittest.main()