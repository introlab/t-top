<template>  
  <face :width="width" :height="height" :drawing="faceDrawer.drawing" :eye-state="displayedEyeState" 
    :mouth-signal="displayMouthState"/>  
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
      displayedEyeState: null,
      currentAnimation: null,
      currentAnimationIndex: 0,
      currentEyeState: null,
      nextEyeStage: null,
      currentFaceTime: 0,

      displayMouthState: null,
    };
  },
  methods: {
    init() {
      this.loopIntervalId = setInterval(function() {
        this.updateCurrentAnimation();
        this.updateStates();
        this.displayedEyeState = this.interpolateDisplayedEyeState();

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

        if (this.currentEyeState === null) {
          this.currentEyeState = this.currentAnimation[0];
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
        this.currentEyeState = this.nextEyeStage;
        this.currentAnimationIndex = (this.currentAnimationIndex + 1) % this.currentAnimation.length;
        this.nextEyeStage = this.currentAnimation[this.currentAnimationIndex];
        this.currentFaceTime = 0;
      }
    },
    interpolateDisplayedEyeState() {
      let displayedEyeState = {};
      for (let property in this.currentEyeState.state) {
        let delta = this.nextEyeStage.state[property] - this.currentEyeState.state[property];
        let offset = delta * this.currentFaceTime / this.nextEyeStage.duration;
        displayedEyeState[property] = this.currentEyeState.state[property] + offset;
      }
      return displayedEyeState;
    },
    updateCurrentMouth() {
      let arrayMouthSignalUp = this.faceDrawer.mouth.arrayMouthSignalUp.map(x => x * this.mouthSignalScale);
      let arrayMouthSignalDown = this.faceDrawer.mouth.arrayMouthSignalDown.map(x => x * this.mouthSignalScale);

      this.displayMouthState = {
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