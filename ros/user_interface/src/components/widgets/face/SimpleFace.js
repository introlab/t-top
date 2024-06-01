const EyeHeight = 250;

function getMouthCurve (mouthSignal, mouthCurveGain) {
  let mouthCurve = [];
  const N = mouthSignal.mouthWidth - 1;
  const A = mouthSignal.mouthWidth / 10 * mouthCurveGain;
  for (let i = 0; i < mouthSignal.mouthWidth; i++) {
    let v = A * Math.cos(0.5 * Math.PI * (i - N / 2) / N) - A;
    mouthCurve.push(v);
  }
  return mouthCurve;
}

function getMouthZigZag (mouthSignal, mouthZigZagGain) {
  let mouthZigZag = [];
  const N = mouthSignal.mouthWidth - 1;
  const A = mouthSignal.mouthWidth / 10 * mouthZigZagGain;
  for (let i = 0; i < mouthSignal.mouthWidth; i++) {
    let v = A * Math.sin(3 * Math.PI * i / N);
    mouthZigZag.push(v);
  }
  return mouthZigZag;
}

export default {
  drawing: {
    updateCanvas: function (canvas, context, state, mouthSignal) {
      if (state === null || state === undefined) {
        state = this.getDefaultState();
      }
      if (mouthSignal === null || mouthSignal === undefined) {
        mouthSignal = this.getDefaultMouthSignal();
      }
      let mouthCurve = getMouthCurve(mouthSignal, state.mouthCurveGain);
      let mouthZigZag = getMouthZigZag(mouthSignal, state.mouthZigZagGain)
      this.drawFrame(canvas, context, state, mouthSignal, mouthCurve, mouthZigZag);
    },
    getDefaultState: function () {
      return {
        leftEyeOutterRadiusX: 80,
        leftEyeOutterRadiusY: 50,
        leftEyeOutterRotation: 0,
        leftEyeInnerRadiusX: 40,
        leftEyeInnerRadiusY: 30,
        leftEyeInnerRotation: 0,
        leftEyeInnerOffsetX: 20,
        leftEyeInnerOffsetY: 10,
        leftEyePupilRadius: 10,
        leftEyePupilOffsetX: 25,
        leftEyePupilOffsetY: 15,

        rightEyeOutterRadiusX: 80,
        rightEyeOutterRadiusY: 50,
        rightEyeOutterRotation: 0,
        rightEyeInnerRadiusX: 40,
        rightEyeInnerRadiusY: 30,
        rightEyeInnerRotation: 0,
        rightEyeInnerOffsetX: -20,
        rightEyeInnerOffsetY: 10,
        rightEyePupilRadius: 10,
        rightEyePupilOffsetX: -25,
        rightEyePupilOffsetY: 15,

        eyeDistance: 220,
        mouthCurveGain: 1,
        mouthZigZagGain: 0
      };
    },
    getDefaultMouthSignal: function () {
      return {
        arrayMouthSignalUp: [0, 14, 25, 33, 38, 42, 45, 47, 48, 49, 49, 49, 48, 47, 45, 42, 38, 33, 25, 14, 0],
        arrayMouthSignalDown: [0, 10, 19, 26, 32, 36, 39, 41, 43, 44, 44, 44, 44, 44, 44, 43, 41, 39, 36, 32, 26, 19, 10, 0],
        mouthHeight: 450,
        mouthWidth: 240
      };
    },
    drawFrame: function (canvas, context, state, mouthSignal, mouthCurve, mouthZigZag) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      this.drawLeftEye(canvas, context, state);
      this.drawRightEye(canvas, context, state);

      context.lineWidth = 5;
      this.drawMouth(canvas, context, mouthSignal, mouthCurve, mouthZigZag);
    },
    drawLeftEye: function (canvas, context, state) {
      let centerX = canvas.width / 2 - state.eyeDistance / 2

      this.drawEye(context, centerX, EyeHeight,
        state.leftEyeOutterRadiusX, state.leftEyeOutterRadiusY, state.leftEyeOutterRotation,
        state.leftEyeInnerRadiusX, state.leftEyeInnerRadiusY, state.leftEyeInnerRotation, state.leftEyeInnerOffsetX, state.leftEyeInnerOffsetY,
        state.leftEyePupilRadius, state.leftEyePupilOffsetX, state.leftEyePupilOffsetY);
    },
    drawRightEye: function (canvas, context, state) {
      let centerX = canvas.width / 2 + state.eyeDistance / 2

      this.drawEye(context, centerX, EyeHeight,
        state.rightEyeOutterRadiusX, state.rightEyeOutterRadiusY, state.rightEyeOutterRotation,
        state.rightEyeInnerRadiusX, state.rightEyeInnerRadiusY, state.rightEyeInnerRotation, state.rightEyeInnerOffsetX, state.rightEyeInnerOffsetY,
        state.rightEyePupilRadius, state.rightEyePupilOffsetX, state.rightEyePupilOffsetY);
    },
    drawEye: function (context, centerX, centerY,
      outterRadiusX, outterRadiusY, outterRotation,
      innerRadiusX, innerRadiusY, innerRotation, innerOffsetX, innerOffsetY,
      pupilRadius, pupilOffsetX, pupilOffsetY) {
      //Draw the inner ellipse
      context.fillStyle = '#000000';
      context.lineWidth = 0;
      context.beginPath();
      context.ellipse(centerX + innerOffsetX, centerY + innerOffsetY, innerRadiusX, innerRadiusY, innerRotation, 0, 2 * Math.PI);
      context.fill();
      //Draw the pupil
      context.globalCompositeOperation = "destination-out"
      context.fillStyle = '#ffffff';
      context.lineWidth = 0;
      context.beginPath();
      context.arc(centerX + pupilOffsetX, centerY + pupilOffsetY, pupilRadius, 0, 2 * Math.PI);
      context.fill();
      context.globalCompositeOperation = "source-over"
      //Draw the outter ellipse
      context.strokeStyle = '#000000';
      context.lineWidth = 5;
      context.beginPath();
      context.ellipse(centerX, centerY, outterRadiusX, outterRadiusY, outterRotation, 0, 2 * Math.PI);
      context.stroke();
    },
    drawMouth: function (canvas, context, mouthSignal, mouthCurve, mouthZigZag) {
      let stepUp = (mouthSignal.mouthWidth - 1) / (mouthSignal.arrayMouthSignalUp.length - 1);
      let stepDown = (mouthSignal.mouthWidth - 1) / (mouthSignal.arrayMouthSignalDown.length - 1);
      let startX = canvas.width / 2 - mouthSignal.mouthWidth / 2;
      context.strokeStyle = '#000000';
      context.lineWidth = 5;
      context.beginPath();
      context.moveTo(startX, mouthSignal.mouthHeight + mouthCurve[0] + mouthZigZag[0]);
      for (let i = 0; i < mouthSignal.arrayMouthSignalDown.length; i++) {
        let x = i * stepDown + startX;
        let y = mouthSignal.mouthHeight + mouthSignal.arrayMouthSignalDown[i] +
          mouthCurve[Math.floor(x - startX)] + mouthZigZag[Math.floor(x - startX)];
        context.lineTo(x, y);
      }
      context.stroke();
      context.beginPath();
      context.moveTo(startX, mouthSignal.mouthHeight + mouthCurve[0] + mouthZigZag[0]);
      for (let i = 0; i < mouthSignal.arrayMouthSignalUp.length; i++) {
        let x = i * stepUp + startX;
        let y = mouthSignal.mouthHeight - mouthSignal.arrayMouthSignalUp[i] +
          mouthCurve[Math.floor(x - startX)] + mouthZigZag[Math.floor(x - startX)];
        context.lineTo(x, y);
      }
      context.stroke();
    }
  },
  faceAnimations: {
    normal: [
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      }
    ],
    sleep: [
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 2,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 0,
          leftEyeInnerRadiusY: 0,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 0,
          leftEyeInnerOffsetY: 0,
          leftEyePupilRadius: 0,
          leftEyePupilOffsetX: 0,
          leftEyePupilOffsetY: 0,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 2,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 0,
          rightEyeInnerRadiusY: 0,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: 0,
          rightEyeInnerOffsetY: 0,
          rightEyePupilRadius: 0,
          rightEyePupilOffsetX: 0,
          rightEyePupilOffsetY: 0,

          eyeDistance: 220,
          mouthCurveGain: 0,
          mouthZigZagGain: 0
        }
      }
    ],
    blink: [
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 2,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 0,
          leftEyeInnerRadiusY: 0,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 0,
          leftEyeInnerOffsetY: 0,
          leftEyePupilRadius: 0,
          leftEyePupilOffsetX: 0,
          leftEyePupilOffsetY: 0,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 2,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 0,
          rightEyeInnerRadiusY: 0,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: 0,
          rightEyeInnerOffsetY: 0,
          rightEyePupilRadius: 0,
          rightEyePupilOffsetX: 0,
          rightEyePupilOffsetY: 0,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 2.0,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      }
    ],
    wink_left: [
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 2,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 0,
          leftEyeInnerRadiusY: 0,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 0,
          leftEyeInnerOffsetY: 0,
          leftEyePupilRadius: 0,
          leftEyePupilOffsetX: 0,
          leftEyePupilOffsetY: 0,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 2.0,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      }
    ],
    wink_right: [
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 2,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 0,
          rightEyeInnerRadiusY: 0,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: 0,
          rightEyeInnerOffsetY: 0,
          rightEyePupilRadius: 0,
          rightEyePupilOffsetX: 0,
          rightEyePupilOffsetY: 0,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 2.0,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      }
    ],
    awe: [
      {
        duration: 0.5,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 50,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 10,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 15,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 50,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 10,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 15,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      },
      {
        duration: 0.5,
        state: {
          leftEyeOutterRadiusX: 64,
          leftEyeOutterRadiusY: 40,
          leftEyeOutterRotation: 0.1,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0.1,
          leftEyeInnerOffsetX: 8,
          leftEyeInnerOffsetY: 4,
          leftEyePupilRadius: 15,
          leftEyePupilOffsetX: 10,
          leftEyePupilOffsetY: 6,

          rightEyeOutterRadiusX: 64,
          rightEyeOutterRadiusY: 40,
          rightEyeOutterRotation: -0.1,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: -0.1,
          rightEyeInnerOffsetX: -8,
          rightEyeInnerOffsetY: 4,
          rightEyePupilRadius: 15,
          rightEyePupilOffsetX: -10,
          rightEyePupilOffsetY: 4,

          eyeDistance: 220,
          mouthCurveGain: 1,
          mouthZigZagGain: 0
        }
      }
    ],
    skeptic: [
      {
        duration: 0.1,
        state: {
          leftEyeOutterRadiusX: 96,
          leftEyeOutterRadiusY: 60,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 48,
          leftEyeInnerRadiusY: 36,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 24,
          leftEyeInnerOffsetY: 12,
          leftEyePupilRadius: 12,
          leftEyePupilOffsetX: 30,
          leftEyePupilOffsetY: 18,

          rightEyeOutterRadiusX: 64,
          rightEyeOutterRadiusY: 40,
          rightEyeOutterRotation: -0.2,
          rightEyeInnerRadiusX: 32,
          rightEyeInnerRadiusY: 24,
          rightEyeInnerRotation: -0.2,
          rightEyeInnerOffsetX: -16,
          rightEyeInnerOffsetY: 8,
          rightEyePupilRadius: 8,
          rightEyePupilOffsetX: -20,
          rightEyePupilOffsetY: 12,

          eyeDistance: 220,
          mouthCurveGain: 0,
          mouthZigZagGain: 0
        }
      }
    ],
    angry: [
      {
        duration: 0.25,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 40,
          leftEyeOutterRotation: 0.5,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 24,
          leftEyeInnerRotation: 0.5,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 8,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 12,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 40,
          rightEyeOutterRotation: -0.5,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 24,
          rightEyeInnerRotation: -0.5,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 8,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 12,

          eyeDistance: 220,
          mouthCurveGain: 0,
          mouthZigZagGain: 0
        }
      }
    ],
    sad: [
      {
        duration: 0.25,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 40,
          leftEyeOutterRotation: -0.5,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 24,
          leftEyeInnerRotation: -0.5,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: -8,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: -12,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 40,
          rightEyeOutterRotation: 0.5,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 24,
          rightEyeInnerRotation: 0.5,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: -8,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: -12,

          eyeDistance: 220,
          mouthCurveGain: -2,
          mouthZigZagGain: 0
        }
      }
    ],
    disgust: [
      {
        duration: 0.25,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 10,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 12,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 0,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 0,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 10,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 12,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 0,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 0,

          eyeDistance: 220,
          mouthCurveGain: 0,
          mouthZigZagGain: 0.5
        }
      }
    ],
    fear: [
      {
        duration: 0.25,
        state: {
          leftEyeOutterRadiusX: 90,
          leftEyeOutterRadiusY: 60,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 45,
          leftEyeInnerRadiusY: 37,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 5,
          leftEyeInnerOffsetY: 3,
          leftEyePupilRadius: 15,
          leftEyePupilOffsetX: 10,
          leftEyePupilOffsetY: 8,

          rightEyeOutterRadiusX: 90,
          rightEyeOutterRadiusY: 60,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 45,
          rightEyeInnerRadiusY: 37,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -5,
          rightEyeInnerOffsetY: 3,
          rightEyePupilRadius: 15,
          rightEyePupilOffsetX: -10,
          rightEyePupilOffsetY: 8,

          eyeDistance: 220,
          mouthCurveGain: 0,
          mouthZigZagGain: 0
        }
      }
    ],
    happy: [
      {
        duration: 0.25,
        state: {
          leftEyeOutterRadiusX: 80,
          leftEyeOutterRadiusY: 40,
          leftEyeOutterRotation: 0,
          leftEyeInnerRadiusX: 40,
          leftEyeInnerRadiusY: 30,
          leftEyeInnerRotation: 0,
          leftEyeInnerOffsetX: 20,
          leftEyeInnerOffsetY: 5,
          leftEyePupilRadius: 10,
          leftEyePupilOffsetX: 25,
          leftEyePupilOffsetY: 10,

          rightEyeOutterRadiusX: 80,
          rightEyeOutterRadiusY: 40,
          rightEyeOutterRotation: 0,
          rightEyeInnerRadiusX: 40,
          rightEyeInnerRadiusY: 30,
          rightEyeInnerRotation: 0,
          rightEyeInnerOffsetX: -20,
          rightEyeInnerOffsetY: 5,
          rightEyePupilRadius: 10,
          rightEyePupilOffsetX: -25,
          rightEyePupilOffsetY: 10,

          eyeDistance: 220,
          mouthCurveGain: 3,
          mouthZigZagGain: 0
        }
      }
    ]
  },
  mouth: {
    //arrayMouthSignalUp length = 21
    //arrayMouthSignalDown length = 24
    arrayMouthSignalUp: [0, 14, 25, 33, 38, 42, 45, 47, 48, 49, 49, 49, 48, 47, 45, 42, 38, 33, 25, 14, 0],
    arrayMouthSignalDown: [0, 10, 19, 26, 32, 36, 39, 41, 43, 44, 44, 44, 44, 44, 44, 43, 41, 39, 36, 32, 26, 19, 10, 0],
    mouthHeight: 450,
    mouthWidth: 240
  }
}
