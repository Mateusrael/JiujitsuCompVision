"""Cached-pose filtering regressions; no images, annotations or models needed."""

from copy import deepcopy
import json
import math
import unittest

from src.Eval.pose_quality import DEFAULT_PROFILE, QUALITY_PROFILES, filter_pose_pair, quality_config


def pose(dx=0, dy=0, confidence=.9, angle=0, scale=1):
    points = [(50,15), (45,12), (55,12), (40,15), (60,15),
              (30,40), (70,40), (20,70), (80,70), (10,100), (90,100),
              (35,100), (65,100), (30,140), (70,140), (25,180), (75,180)]
    result=[]
    for x,y in points:
        x,y=(x-50)*scale,(y-100)*scale
        result.append([400+dx+x*math.cos(angle)-y*math.sin(angle),
                       400+dy+x*math.sin(angle)+y*math.cos(angle), confidence])
    return result


class PoseQualityTests(unittest.TestCase):
    def filter(self, poses, detector="yolo", profile=DEFAULT_PROFILE):
        result=filter_pose_pair(poses,detector=detector,profile=profile)
        json.dumps(result,allow_nan=False)
        self.assertIn(len(result["poses"]),(0,2))
        audit=result["quality_filter"]
        self.assertEqual(audit["profile"],profile)
        self.assertEqual(audit["detector"],detector)
        self.assertEqual(audit["input_pose_count"],len(poses))
        self.assertEqual(len(audit["pose_quality"]),len(poses))
        return result

    def test_reliable_pair_preserves_original_pose_order_and_coordinates(self):
        inputs=[pose(),pose(100)]
        original=deepcopy(inputs)
        result=self.filter(inputs)
        self.assertTrue(result["quality_filter"]["accepted"])
        self.assertEqual(result["quality_filter"]["reason"],"accepted")
        self.assertEqual(result["poses"],inputs)
        self.assertEqual(inputs,original)

    def test_zero_and_one_pose_are_retained_as_audited_rejections(self):
        for poses in [[],[pose()]]:
            result=self.filter(poses)
            self.assertEqual(result["poses"],[])
            self.assertEqual(result["quality_filter"]["reason"],"missing_pair")

    def test_one_joint_per_pose_and_zero_filled_poses_do_not_qualify(self):
        one_joint=[[0,0,0]]*16+[pose()[16]]
        for weak in [one_joint,[[0,0,0]]*17]:
            result=self.filter([weak,pose(100)])
            self.assertEqual(result["poses"],[])
            self.assertEqual(result["quality_filter"]["reason"],"unreliable_pose")
            self.assertIn("insufficient_body_joints",result["quality_filter"]["pose_quality"][0]["reasons"])

    def test_v1_preserves_original_torso_and_lower_limb_requirements(self):
        for removed,reason in [([5,6],"insufficient_torso_joints"),
                               ([13,14,15],"insufficient_lower_limb_joints")]:
            first=pose()
            for i in removed:first[i]=[0,0,0]
            result=self.filter([first,pose(100)], profile="cached-pose-quality-v1")
            self.assertFalse(result["quality_filter"]["accepted"])
            self.assertIn(reason,result["quality_filter"]["pose_quality"][0]["reasons"])

    def test_v2_keeps_partial_poses_with_two_torso_joints_or_hidden_lower_limbs(self):
        for removed in ((5,6), (11,12), (13,14,15), (13,14,15,16)):
            first=pose()
            for index in removed:first[index]=[0,0,0]
            original=deepcopy(first)
            for detector in ("yolo", "vitpose"):
                with self.subTest(removed=removed, detector=detector):
                    old=self.filter([first,pose(100)], detector, "cached-pose-quality-v1")
                    new=self.filter([first,pose(100)], detector)
                    self.assertFalse(old["quality_filter"]["accepted"])
                    self.assertTrue(new["quality_filter"]["accepted"])
                    self.assertEqual(new["poses"][0], original)
            self.assertEqual(first,original)

    def test_v2_still_requires_two_torso_joints_and_eight_body_joints(self):
        for removed,reason in (((5,6,11),"insufficient_torso_joints"),
                               ((9,13,14,15,16),"insufficient_body_joints")):
            first=pose()
            for index in removed:first[index]=[0,0,0]
            result=self.filter([first,pose(100)])
            self.assertFalse(result["quality_filter"]["accepted"])
            self.assertIn(reason,result["quality_filter"]["pose_quality"][0]["reasons"])

    def test_relaxation_does_not_admit_duplicate_partial_skeletons(self):
        first,second=pose(),pose(1)
        for index in (13,14,15,16):
            first[index]=second[index]=[0,0,0]
        result=self.filter([first,second])
        self.assertEqual(result["quality_filter"]["reason"],"duplicate_skeletons")

    def test_missing_face_is_allowed_when_body_is_reliable(self):
        first=pose()
        for i in range(5):first[i][2]=.05
        result=self.filter([first,pose(100)])
        self.assertTrue(result["quality_filter"]["accepted"])
        self.assertEqual(result["poses"][0][:5],[[0,0,0]]*5)

    def test_bad_ankle_is_rejected_before_pair_normalization(self):
        first=pose();first[15]=[1450,1000,.95]
        result=self.filter([first,pose(100)])
        self.assertFalse(result["quality_filter"]["accepted"])
        reasons=result["quality_filter"]["pose_quality"][0]["reasons"]
        self.assertTrue({"body_joint_outlier","body_segment_outlier"}.intersection(reasons))

    def test_collapsed_pose_or_torso_is_rejected(self):
        collapsed=[[400,400,.9]]*17
        torso=pose()
        for joint in (5,6,11,12):torso[joint]=[400,400,.95]
        for first,reason in [(collapsed,"degenerate_body_geometry"),
                             (torso,"degenerate_torso_geometry")]:
            result=self.filter([first,pose(100)])
            self.assertEqual(result["poses"],[])
            self.assertIn(reason,result["quality_filter"]["pose_quality"][0]["reasons"])

    def test_high_confidence_face_outlier_cannot_distort_normalization(self):
        for joint in range(5):
            first=pose();first[joint]=[1400,1000,.99]
            result=self.filter([first,pose(100)])
            self.assertEqual(result["poses"],[])
            self.assertIn("head_joint_outlier",result["quality_filter"]["pose_quality"][0]["reasons"])

    def test_duplicate_skeletons_are_not_two_people(self):
        for profile in QUALITY_PROFILES:
            result=self.filter([pose(),pose(1)],profile=profile)
            self.assertEqual(result["poses"],[])
            self.assertEqual(result["quality_filter"]["reason"],"duplicate_skeletons")
            self.assertTrue(all("duplicate_skeleton" in item["reasons"]
                                for item in result["quality_filter"]["pose_quality"]))

    def test_overlapping_distinct_skeletons_are_kept(self):
        result=self.filter([pose(),pose(angle=math.pi/2)])
        self.assertTrue(result["quality_filter"]["accepted"])

    def test_arbitrary_rotation_and_input_permutation_preserve_pair(self):
        first,second=pose(angle=2.1),pose(95,angle=-.7)
        normal=self.filter([first,second])
        reverse=self.filter([second,first])
        self.assertTrue(normal["quality_filter"]["accepted"])
        self.assertEqual(normal["poses"],list(reversed(reverse["poses"])))

    def test_no_image_dimensions_or_pair_distance_assumption(self):
        # Filtering sees cached coordinates only; it cannot invent image bounds
        # or select different people based on unavailable person detections.
        result=self.filter([pose(5000),pose(8000,dy=4000)])
        self.assertTrue(result["quality_filter"]["accepted"])

    def test_detector_thresholds_and_nonfinite_negative_joints(self):
        first=pose(confidence=.4)
        self.assertFalse(self.filter([first,pose(100)])["quality_filter"]["accepted"])
        self.assertTrue(self.filter([first,pose(100)],"vitpose")["quality_filter"]["accepted"])
        first=pose()
        first[0]=[float("nan"),400,.9]
        first[1]=[-1,400,.9]
        first[2]=[400,float("inf"),.9]
        first[3]=[400,400,float("nan")]
        first[4][2]=2
        original=deepcopy(first)
        result=self.filter([first,pose(100)])
        self.assertTrue(result["quality_filter"]["accepted"])
        self.assertEqual(result["poses"][0][:4],[[0,0,0]]*4)
        self.assertEqual(result["poses"][0][4][2],1)
        self.assertEqual(first[4],original[4])
        self.assertTrue(math.isnan(first[0][0]))

    def test_quality_config_is_fresh_and_invalid_input_shapes_fail(self):
        config=quality_config("yolo");config["min_body_joints"]=0
        self.assertEqual(quality_config("yolo")["min_body_joints"],8)
        with self.assertRaises(ValueError):quality_config("unknown")
        with self.assertRaises(ValueError):quality_config("yolo", "unknown-profile")
        old=quality_config("vitpose", "cached-pose-quality-v1")
        new=quality_config("vitpose")
        changed={key for key in old if old[key] != new[key]}
        self.assertEqual(changed,{"profile","min_torso_joints","min_lower_limb_joints"})
        for invalid in [None,{},[pose()]*3,[None],[[[0,0,.9]]],
                        [[[True,0,.9]]*17]]:
            with self.assertRaises(ValueError):filter_pose_pair(invalid,detector="yolo")


if __name__=="__main__":
    unittest.main()
