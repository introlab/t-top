<template>
  <canvas ref="canvas" :width="width" :height="height"/>
</template>

<script>
export default {
  name: 'Face',
  props: ['width', 'height', 'drawing', 'eyeState', 'mouthSignal'],
  data() {
    return {
      loopIntervalId: null,
      canvas: null,
      context: null,
      mouthCurve: null
    };
  },
  methods: {
    updateCanvas() {
      this.drawing.updateCanvas(this.canvas, this.context, this.eyeState, this.mouthSignal);
    }
  },
  watch: {
    width() {
      this.updateCanvas();
    },
    height() {
      this.updateCanvas();
    },
    eyeHeight() {
      this.updateCanvas();
    },
    eyeState() {
      this.updateCanvas();
    },
    mouthSignal() {
      this.updateCanvas();
    }
  },
  mounted() {
    this.canvas = this.$refs.canvas;
    this.context = this.context = this.canvas.getContext('2d');
  },
  destroyed() {
    clearInterval(this.loopIntervalId);
  }
}
</script>

<style scoped>
canvas {
  background-color: #ffffff
}
</style>
