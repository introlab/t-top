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
  mounted() {
    if (this.ros === null) {
      this.ros = new ROSLIB.Ros({
        url: Constants.RosUrl
      });
    }
  }
}
</script>

<style>
</style>