<template>
  <div>
    <animated-face :width="faceWidth" :height="faceHeight" :face-drawer="faceDrawer" :animation="animation" :mouth-signal-scale="mouthSignalScale"/>
  </div>
</template>

<script>
import ROSLIB from 'roslib';

import Constants from '../../Constants';

import AnimatedFace from '../widgets/face/AnimatedFace';
import SimpleFace from '../widgets/face/SimpleFace';
import MaleFace from '../widgets/face/MaleFace';


function getFaceDrawer() {
  if (Constants.FaceNumber == 1) {
    return SimpleFace;
  }
  if (Constants.FaceNumber == 2) {
    return MaleFace;
  }
  return SimpleFace;
}

export default {
  name: 'FacePage',
  components: {
    AnimatedFace
  },
  props: ['ros'],
  data() {
    return {
      faceWidth: Constants.FaceWidth.toString() + 'px',
      faceHeight: Constants.FaceHeight.toString() + 'px',

      faceDrawer: SimpleFace,

      animation: null,
      animationSub: null,
      mouthSignalScale: 0.0,
      mouthSignalScaleSub: null,
    };
  },
  methods: {
    animationSubCallback(message) {
      if (Object.keys(this.faceDrawer.faceAnimations).includes(message.data)) {
        this.animation = this.faceDrawer.faceAnimations[message.data];
      }
      else {
        this.animation = this.faceDrawer.faceAnimations['normal'];
      }
    },
    mouthSignalScaleCallback(message) {
      this.mouthSignalScale = message.data
    }
  },
  watch: {
    ros() {
      if (this.ros === null || this.ros === undefined) {
        return;
      }

      if (this.animationSub === null) {
        this.animationSub = new ROSLIB.Topic({
          ros : this.ros,
          name : '/face/animation',
          messageType : 'std_msgs/String'
        });

        this.animationSub.subscribe(this.animationSubCallback);
      }

      if (this.mouthSignalScaleSub === null) {
        this.mouthSignalScaleSub = new ROSLIB.Topic({
          ros : this.ros,
          name : '/face/mouth_signal_scale',
          messageType : 'std_msgs/Float32'
        });

        this.mouthSignalScaleSub.subscribe(this.mouthSignalScaleCallback);
      }
    }
  },
  mounted() {
    this.faceDrawer = getFaceDrawer();
  },
  destroyed() {
    if (this.animationSub !== null) {
      this.animationSub.unsubscribe();
    }

    if (this.mouthSignalSub !== null) {
      this.mouthSignalSub.unsubscribe();
    }
  }
}
</script>

<style scoped>
</style>
