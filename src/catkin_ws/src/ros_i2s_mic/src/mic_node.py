#!/usr/bin/env python3

import rospy
import alsaaudio

from std_msgs.msg import UInt8MultiArray

class I2SMicNode:
    def __init__(self):

        self.device = rospy.get_param("~device", "hw:1,0")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))
        self.period_size = int(rospy.get_param("~period_size", 512))

        self.publisher = rospy.Publisher(
            "/audio/raw",
            UInt8MultiArray,
            queue_size=2
        )

        self.mic = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NORMAL,
            device=self.device
        )

        self.mic.setchannels(self.channels)
        self.mic.setrate(self.sample_rate)
        self.mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self.mic.setperiodsize(self.period_size)

        rospy.loginfo("I2S mic started")

    def run(self):

        rate = rospy.Rate(100)

        while not rospy.is_shutdown():

            length, data = self.mic.read()

            if length > 0:

                msg = UInt8MultiArray()
                msg.data = bytearray(data)

                self.publisher.publish(msg)

            rate.sleep()


def main():
    rospy.init_node("ros_i2s_mic")

    node = I2SMicNode()
    node.run()


if __name__ == "__main__":
    main()
