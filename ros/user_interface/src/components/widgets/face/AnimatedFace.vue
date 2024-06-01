<template>
  <face :width="width" :height="height" :drawing="faceDrawer.drawing" :state="displayedState"
    :mouth-signal="displayedMouthState"/>
</template>

<script>
import Constants from '../../../Constants';

import Face from './Face'

const StepDuration = 1 / Constants.CanvasRefreshRate;

export default {
  name: 'AnimatedFace',
  components: {
    Face
  },
  props: ['width', 'height', 'faceDrawer', 'animation', 'mouthSignalScale'],

  data() {
    return {
      displayedState: null,
      currentAnimation: null,
      currentAnimationIndex: 0,
      currentState: null,
      nextEyeStage: null,
      currentFaceTime: 0,

      displayedMouthState: null,
    };
  },
  methods: {
    init() {
      this.loopIntervalId = setInterval(function() {
        this.updateCurrentAnimation();
        this.updateStates();
        this.displayedState = this.interpolateDisplayedState();

        this.updateCurrentMouth();

        this.currentFaceTime += StepDuration;
      }.bind(this), 1000 / Constants.CanvasRefreshRate);
    },
    updateCurrentAnimation() {
      let animation = this.animation;
      if (animation === null || animation === undefined) {
        animation = this.faceDrawer.faceAnimations['normal'];
      }
      if (this.currentAnimation === null || this.currentAnimation !== animation) {
        this.currentAnimation = animation;
        this.currentFaceTime = 0;

        if (this.currentState === null) {
          this.currentState = this.currentAnimation[0];
          this.currentAnimationIndex = 1 % this.currentAnimation.length;
          this.nextEyeStage = this.currentAnimation[this.currentAnimationIndex];
        }
        else {
          this.nextEyeStage = this.currentAnimation[0];
          this.currentAnimationIndex = 0;
        }
      }
    },
    updateStates() {
      if (this.currentFaceTime > this.nextEyeStage.duration) {
        this.currentState = this.nextEyeStage;
        this.currentAnimationIndex = (this.currentAnimationIndex + 1) % this.currentAnimation.length;
        this.nextEyeStage = this.currentAnimation[this.currentAnimationIndex];
        this.currentFaceTime = 0;
      }
    },
    interpolateDisplayedState() {
      let displayedState = {};
      for (let property in this.currentState.state) {
        let delta = this.nextEyeStage.state[property] - this.currentState.state[property];
        let offset = delta * this.currentFaceTime / this.nextEyeStage.duration;
        displayedState[property] = this.currentState.state[property] + offset;
      }
      return displayedState;
    },
    updateCurrentMouth() {
      let arrayMouthSignalUp = this.faceDrawer.mouth.arrayMouthSignalUp.map(x => x * this.mouthSignalScale);
      let arrayMouthSignalDown = this.faceDrawer.mouth.arrayMouthSignalDown.map(x => x * this.mouthSignalScale);

      this.displayedMouthState = {
        arrayMouthSignalUp,
        arrayMouthSignalDown,
        mouthHeight: this.faceDrawer.mouth.mouthHeight,
        mouthWidth: this.faceDrawer.mouth.mouthWidth
      }
    }
  },
  mounted() {
    this.init();
  },
  destroyed() {
    clearInterval(this.loopIntervalId);
  }
}
</script>

<style scoped>
</style>
