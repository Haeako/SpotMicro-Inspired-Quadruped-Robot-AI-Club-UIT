#!/usr/bin/env python3
import unittest
import rospy
from std_msgs.msg import String

class TestRingBuffer(unittest.TestCase):

    def setUp(self):
        rospy.init_node('test_ringbuffer', anonymous=True)
        self.pub = rospy.Publisher('/test_topic', String, queue_size=10)
        self.sub = rospy.Subscriber('/test_topic', String, self.callback)
        self.received_message = None

    def callback(self, msg):
        self.received_message = msg.data

    def test_publish_and_subscribe(self):
        test_message = "Hello, ROS!"
        self.pub.publish(test_message)
        rospy.sleep(1)  # Allow time for the message to be received
        self.assertEqual(self.received_message, test_message)

if __name__ == '__main__':
    import rosunit
    rosunit.unitrun('spot_keyword_spotting', 'test_ringbuffer', TestRingBuffer)