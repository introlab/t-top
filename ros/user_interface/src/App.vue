<template>
  <div id="app">
    <router-view :ros="ros"/>
  </div>
</template>

<script>
import ROSLIB from 'roslib'
import Constants from './Constants'

export default {
  name: 'app',
  data() {
    return {
      ros: null
    }
  },
  methods: {
    reloadLater() {
      setTimeout(() => location.reload(), Constants.ReloadInterval);
    }
  },
  mounted() {
    if (this.ros === null) {
      this.ros = new ROSLIB.Ros({
        url: Constants.RosUrl
      });

      this.ros.on('error', this.reloadLater);
      this.ros.on('close', this.reloadLater);
    }
  }
}
</script>

<style>
</style>
