import Constants from '../../../Constants';

function getMouthCurve (n, a) {
  let mouthCurve = [];
  for (let i = 0; i < n; i++) {
    let v = a * Math.cos(0.5 * Math.PI * (i - n) / n);
    mouthCurve.push(v);
  }
  return mouthCurve.map(x => x - mouthCurve[0]);
}

export default {
  drawing: {
    updateCanvas: function (canvas, context, state, mouthSignal) {
      if (state === null || state === undefined) {
        state = this.getDefaultState();
      }
      if (mouthSignal === null || mouthSignal === undefined) {
        mouthSignal = this.getDefaultMouthSignal(canvas);
      }
      this.drawFrame(canvas, context, state, mouthSignal);
    },
    getDefaultState: function () {
      return {
        x1_leftEye: 116,
        x2_leftEye: 126,
        x3_leftEye: 142,
        x4_leftEye: 158,
        x5_leftEye: 172,
        x6_leftEye: 182,
        x7_leftEye: 188,

        y1_leftEyeUp: 384,
        y2_leftEyeUp: 373,
        y3_leftEyeUp: 370,
        y4_leftEyeUp: 368,
        y5_leftEyeUp: 373,
        y6_leftEyeUp: 377,
        y7_leftEyeUp: 384,

        y2_leftEyeDown: 392,
        y3_leftEyeDown: 394,
        y4_leftEyeDown: 396,
        y5_leftEyeDown: 391,
        y6_leftEyeDown: 385,

        x1_rightEye: Constants.FaceWidth - 116,
        x2_rightEye: Constants.FaceWidth - 126,
        x3_rightEye: Constants.FaceWidth - 142,
        x4_rightEye: Constants.FaceWidth - 158,
        x5_rightEye: Constants.FaceWidth - 172,
        x6_rightEye: Constants.FaceWidth - 182,
        x7_rightEye: Constants.FaceWidth - 188,

        y1_rightEyeUp: 384,
        y2_rightEyeUp: 373,
        y3_rightEyeUp: 370,
        y4_rightEyeUp: 368,
        y5_rightEyeUp: 373,
        y6_rightEyeUp: 377,
        y7_rightEyeUp: 384,

        y2_rightEyeDown: 392,
        y3_rightEyeDown: 394,
        y4_rightEyeDown: 396,
        y5_rightEyeDown: 391,
        y6_rightEyeDown: 385,


        x_leftPupil: 152,
        y_leftPupil: 382,
        r_leftPupil: 7,
        x_leftIris: 152,
        y_leftIris: 382,
        r_leftIris: 17,

        x_rightPupil: Constants.FaceWidth - 152,
        y_rightPupil: 382,
        r_rightPupil: 7,
        x_rightIris: Constants.FaceWidth - 152,
        y_rightIris: 382,
        r_rightIris: 17,


        x1_leftEyebrow: 92,
        x2_leftEyebrow: 120,
        x3_leftEyebrow: 200,

        y1_leftEyebrowUp: 357,
        y2_leftEyebrowUp: 332,
        y3_leftEyebrowUp: 349,
        y1_leftEyebrowDown: 359,
        y2_leftEyebrowDown: 338,
        y3_leftEyebrowDown: 359,

        x1_rightEyebrow: Constants.FaceWidth - 92,
        x2_rightEyebrow: Constants.FaceWidth - 120,
        x3_rightEyebrow: Constants.FaceWidth - 200,

        y1_rightEyebrowUp: 357,
        y2_rightEyebrowUp: 332,
        y3_rightEyebrowUp: 349,
        y1_rightEyebrowDown: 359,
        y2_rightEyebrowDown: 338,
        y3_rightEyebrowDown: 359,

        upperMouthCurveA: 0,
        lowerMouthCurveA: 0
      };
    },
    getDefaultMouthSignal: function () {
      return {
        arrayMouthSignalUp: [164, 178, 188, 197, 208, 218, Constants.FaceWidth/2, 570, 568, 564, 561, 562, 564, 564, 164, 178, 188, 197, 208, 218, Constants.FaceWidth/2, 570, 571, 574, 577, 578, 580, 580],
        arrayMouthSignalDown: [164, 178, 188, 197, 208, 218, Constants.FaceWidth/2, 570, 571, 574, 577, 578, 580, 580, 164, 178, 188, 197, 208, 218, Constants.FaceWidth/2, 570, 584, 588, 593, 596, 599, 600],
        mouthHeight: 0,
        mouthWidth: 0
      };
    },
    drawFrame: function (canvas, context, state, mouthSignal) {
      context.clearRect(0, 0, Constants.FaceWidth, canvas.height);
      this.draw_sweater(canvas, context);
      this.draw_neck(canvas, context);
      this.draw_hair_outline(canvas, context);
      this.draw_inner_hair(canvas, context);
      this.draw_ears_outline(canvas, context);
      this.draw_inner_ears(canvas, context);
      this.draw_face_contour(canvas, context, mouthSignal.mouthHeight);
      this.draw_nose(canvas, context);
      this.draw_dark_circles(canvas, context);

      this.draw_left_eye(canvas, context, state);
      this.draw_right_eye(canvas, context, state);

      this.draw_mouth(canvas, context, mouthSignal, state);
    },
    draw_sweater: function (canvas, context) {
      const x1 = 100, y1 = 624, x2 = 48, y2 = 704, x3 = 0, y3 = 740, x4 = 0, y4 = canvas.height, x5 = Constants.FaceWidth/2, y5 = canvas.height, x6 = 55, y6 = 740, x7 = 72, y7 = canvas.height;

      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1));
      context.lineTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2));
      context.lineTo(this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.lineTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4));
      context.lineTo(this.ratioX(canvas, x5), this.ratioY(canvas, y5));

      // Symmetry
      context.lineTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4));
      context.lineTo(this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.lineTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2));
      context.lineTo(this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));
      context.fillStyle = "#0000CD";
      context.fill();

      context.moveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2));
      context.quadraticCurveTo(this.ratioX(canvas, x6), this.ratioY(canvas, y6), this.ratioX(canvas, x7), this.ratioY(canvas, y7));

      // Symmetry
      context.moveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x6), this.ratioY(canvas, y6), this.ratioX(canvas, Constants.FaceWidth-x7), this.ratioY(canvas, y7));
      context.strokeStyle = "#000000";
      context.lineWidth = 5;
      context.stroke();
    },
    draw_neck: function (canvas, context) {
      const x1 = 92, y1 = 557, x2 = 95, y2 = 592, x3 = 100, y3 = 672, x4 = 110, y4 = 720, x5 = 156, y5 = canvas.height, x6 = Constants.FaceWidth/2, y6 = canvas.height;
      const x7 = 210, y7 = canvas.height, x8 = 200, y8 = 748, x10 = 190, y10 = 725;

      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1))
      context.quadraticCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4), this.ratioX(canvas, x5), this.ratioY(canvas, y5));
      context.lineTo(this.ratioX(canvas, x6), this.ratioY(canvas, y6));

      // Symmetry
      context.lineTo(this.ratioX(canvas, Constants.FaceWidth-x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4), this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));

      context.fillStyle = "#DEB887";
      context.fill();
      context.lineWidth = 3;
      context.stroke();

      context.moveTo(this.ratioX(canvas, x7), this.ratioY(canvas, y7));
      context.quadraticCurveTo(this.ratioX(canvas, x8), this.ratioY(canvas, y8), this.ratioX(canvas, x10), this.ratioY(canvas, y10));
      context.lineWidth = 2;
      context.stroke();
    },
    draw_hair_outline: function (canvas, context) {
      const x1 = 64, y1 = 352, x2 = 56, y2 = 288, x3 = 52, y3 = 240, x4 = 50, y4 = 200, x5 = 60, y5 = 176, x6 = 76, y6 = 148, x7 = 92, y7 = 136;
      const x8 = 60, y8 = 156, x9 = 40, y9 = 152, x10 = 64, y10 = 148, x11 = 84, y11 = 128, x12 = 72, y12 = 132, x13 = 60, y13 = 144, x14 = 65, y14 = 130, x15 = 96, y15 = 112;
      const x16 = 72, y16 = 116, x17 = 52, y17 = 122, x18 = 84, y18 = 96, x19 = 120, y19 = 100, x20 = 90, y20 = 90, x21 = 76, y21 = 60, x22 = 100, y22 = 84, x23 = 144, y23 = 88;
      const x24 = 112, y24 = 80, x25 = 96, y25 = 64, x26 = 128, y26 = 72, x27 = 160, y27 = 76, x28 = 140, y28 = 60, x29 = 132, y29 = 40, x30 = 152, y30 = 64, x31 = 200, y31 = 72;
      const x32 = 180, y32 = 64, x33 = 168, y33 = 52, x34 = 196, y34 = 60, x35 = 224, y35 = 68, x36 = 200, y36 = 48, x37 = 188, y37 = 20, x38 = 216, y38 = 48, x39 = 252, y39 = 68;
      const x40 = 240, y40 = 40, x41 = 244, y41 = 16, x42 = 248, y42 = 48, x43 = 276, y43 = 76, x44 = 260, y44 = 52, x45 = 256, y45 = 28, x46 = 264, y46 = 48, x47 = 288, y47 = 62;
      const x48 = 370, y48 = 90, x49 = 390, y49 = 152;

      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4), this.ratioX(canvas, x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, x6), this.ratioY(canvas, y6), this.ratioX(canvas, x7), this.ratioY(canvas, y7));

      // Hair tip
      context.quadraticCurveTo(this.ratioX(canvas, x8), this.ratioY(canvas, y8), this.ratioX(canvas, x9), this.ratioY(canvas, y9));
      context.quadraticCurveTo(this.ratioX(canvas, x10), this.ratioY(canvas, y10), this.ratioX(canvas, x11), this.ratioY(canvas, y11));
      context.quadraticCurveTo(this.ratioX(canvas, x12), this.ratioY(canvas, y12), this.ratioX(canvas, x13), this.ratioY(canvas, y13));
      context.quadraticCurveTo(this.ratioX(canvas, x14), this.ratioY(canvas, y14), this.ratioX(canvas, x15), this.ratioY(canvas, y15));
      context.quadraticCurveTo(this.ratioX(canvas, x16), this.ratioY(canvas, y16), this.ratioX(canvas, x17), this.ratioY(canvas, y17));
      context.quadraticCurveTo(this.ratioX(canvas, x18), this.ratioY(canvas, y18), this.ratioX(canvas, x19), this.ratioY(canvas, y19));
      context.quadraticCurveTo(this.ratioX(canvas, x20), this.ratioY(canvas, y20), this.ratioX(canvas, x21), this.ratioY(canvas, y21));
      context.quadraticCurveTo(this.ratioX(canvas, x22), this.ratioY(canvas, y22), this.ratioX(canvas, x23), this.ratioY(canvas, y23));
      context.quadraticCurveTo(this.ratioX(canvas, x24), this.ratioY(canvas, y24), this.ratioX(canvas, x25), this.ratioY(canvas, y25));
      context.quadraticCurveTo(this.ratioX(canvas, x26), this.ratioY(canvas, y26), this.ratioX(canvas, x27), this.ratioY(canvas, y27));
      context.quadraticCurveTo(this.ratioX(canvas, x28), this.ratioY(canvas, y28), this.ratioX(canvas, x29), this.ratioY(canvas, y29));
      context.quadraticCurveTo(this.ratioX(canvas, x30), this.ratioY(canvas, y30), this.ratioX(canvas, x31), this.ratioY(canvas, y31));
      context.quadraticCurveTo(this.ratioX(canvas, x32), this.ratioY(canvas, y32), this.ratioX(canvas, x33), this.ratioY(canvas, y33));
      context.quadraticCurveTo(this.ratioX(canvas, x34), this.ratioY(canvas, y34), this.ratioX(canvas, x35), this.ratioY(canvas, y35));
      context.quadraticCurveTo(this.ratioX(canvas, x36), this.ratioY(canvas, y36), this.ratioX(canvas, x37), this.ratioY(canvas, y37));
      context.quadraticCurveTo(this.ratioX(canvas, x38), this.ratioY(canvas, y38), this.ratioX(canvas, x39), this.ratioY(canvas, y39));
      context.quadraticCurveTo(this.ratioX(canvas, x40), this.ratioY(canvas, y40), this.ratioX(canvas, x41), this.ratioY(canvas, y41));
      context.quadraticCurveTo(this.ratioX(canvas, x42), this.ratioY(canvas, y42), this.ratioX(canvas, x43), this.ratioY(canvas, y43));
      context.quadraticCurveTo(this.ratioX(canvas, x44), this.ratioY(canvas, y44), this.ratioX(canvas, x45), this.ratioY(canvas, y45));
      context.quadraticCurveTo(this.ratioX(canvas, x46), this.ratioY(canvas, y46), this.ratioX(canvas, x47), this.ratioY(canvas, y47));
      context.quadraticCurveTo(this.ratioX(canvas, x48), this.ratioY(canvas, y48), this.ratioX(canvas, x49), this.ratioY(canvas, y49));

      // Symmetry
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x6), this.ratioY(canvas, y6), this.ratioX(canvas, Constants.FaceWidth-x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4), this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));

      context.fillStyle = "#A0522D";
      context.fill();
      context.lineWidth = 3;
      context.stroke();

    },
    draw_inner_hair: function (canvas, context) {
      const x1 = 72, y1 = 184, x2 = 110, y2 = 180, x3 = 148, y3 = 212, x4 = 75, y4 = 180, x5 = 108, y5 = 172, x6 = 140, y6 = 190;
      const x7 = 93, y7 = 136, x8 = 200, y8 = 75, x9 = 280, y9 = 160;
      const x10 = 386, y10 = 150, x11 = 320, y11 = 170, x12 = 300, y12 = 224, x13 = 344, y13 = 172, x14 = 360, y14 = 176, x15 = 372, y15 = 185, x16 = 344, y16 = 148, x17 = 320, y17 = 132;
      const x18 = 330, y18 = 180, x19 = 348, y19 = 180, x20 = 376, y20 = 200;
      const x21 = 312, y21 = 200, x22 = 308, y22 = 185, x23 = 295, y23 = 176, x24 = 315, y24 = 192, x25 = 306, y25 = 175;

      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x3), this.ratioY(canvas, y3));

      context.moveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4));
      context.quadraticCurveTo(this.ratioX(canvas, x5), this.ratioY(canvas, y5), this.ratioX(canvas, x6), this.ratioY(canvas, y6));

      context.moveTo(this.ratioX(canvas, x7), this.ratioY(canvas, y7));
      context.quadraticCurveTo(this.ratioX(canvas, x8), this.ratioY(canvas, y8), this.ratioX(canvas, x9), this.ratioY(canvas, y9));

      context.moveTo(this.ratioX(canvas, x10), this.ratioY(canvas, y10));
      context.quadraticCurveTo(this.ratioX(canvas, x11), this.ratioY(canvas, y11), this.ratioX(canvas, x12), this.ratioY(canvas, y12));

      context.moveTo(this.ratioX(canvas, x13), this.ratioY(canvas, y13));
      context.quadraticCurveTo(this.ratioX(canvas, x14), this.ratioY(canvas, y14), this.ratioX(canvas, x15), this.ratioY(canvas, y15));

      context.moveTo(this.ratioX(canvas, x13), this.ratioY(canvas, y13));
      context.quadraticCurveTo(this.ratioX(canvas, x16), this.ratioY(canvas, y16), this.ratioX(canvas, x17), this.ratioY(canvas, y17));

      context.moveTo(this.ratioX(canvas, x18), this.ratioY(canvas, y18));
      context.quadraticCurveTo(this.ratioX(canvas, x19), this.ratioY(canvas, y19), this.ratioX(canvas, x20), this.ratioY(canvas, y20));

      context.moveTo(this.ratioX(canvas, x21), this.ratioY(canvas, y21));
      context.quadraticCurveTo(this.ratioX(canvas, x22), this.ratioY(canvas, y22), this.ratioX(canvas, x23), this.ratioY(canvas, y23));

      context.moveTo(this.ratioX(canvas, x21), this.ratioY(canvas, y21));
      context.quadraticCurveTo(this.ratioX(canvas, x24), this.ratioY(canvas, y24), this.ratioX(canvas, x25), this.ratioY(canvas, y25));

      context.lineWidth = 4;
      context.strokeStyle = "#8B4513";
      context.lineCap = "round";
      context.stroke();
    },
    draw_ears_outline: function (canvas, context) {
      const x1 = 68, y1 = 380, x2 = 52, y2 = 352, x3 = 40, y3 = 342, x4 = 31, y4 = 336, x5 = 25, y5 = 350, x6 = 21, y6 = 368, x7 = 27, y7 = 404, x8 = 38, y8 = 447, x9 = 47, y9 = 458, x10 = 58, y10 = 470, x11 = 72, y11 = 472;

      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4), this.ratioX(canvas, x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, x6), this.ratioY(canvas, y6), this.ratioX(canvas, x7), this.ratioY(canvas, y7));
      context.quadraticCurveTo(this.ratioX(canvas, x8), this.ratioY(canvas, y8), this.ratioX(canvas, x9), this.ratioY(canvas, y9));
      context.quadraticCurveTo(this.ratioX(canvas, x10), this.ratioY(canvas, y10), this.ratioX(canvas, x11), this.ratioY(canvas, y11));

      // Symmetry
      context.moveTo(this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4), this.ratioX(canvas, Constants.FaceWidth-x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x6), this.ratioY(canvas, y6), this.ratioX(canvas, Constants.FaceWidth-x7), this.ratioY(canvas, y7));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x8), this.ratioY(canvas, y8), this.ratioX(canvas,Constants.FaceWidth- x9), this.ratioY(canvas, y9));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x10), this.ratioY(canvas, y10), this.ratioX(canvas, Constants.FaceWidth-x11), this.ratioY(canvas, y11));

      context.fillStyle = "#DEB887";
      context.fill();
      context.lineWidth = 3;
      context.strokeStyle = "#000000";
      context.stroke();
    },
    draw_inner_ears: function (canvas, context) {
      const x1 = 68, y1 = 404, x2 = 47, y2 = 368, x3 = 32, y3 = 352, x4 = 28, y4 = 350, x5 = 24, y5 = 352, x6 = 26, y6 = 402, x7 = 70, y7 = 456, x8 = 57, y8 = 455, x9 = 45, y9 = 440;
      const x10 = 27, y10 = 358, x11 = 28, y11 = 370;
      const x12 = 40, y12 = 420, x13 = 70, y13 = 432;

      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1));
      context.bezierCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.bezierCurveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4), this.ratioX(canvas, x4), this.ratioY(canvas, y4), this.ratioX(canvas, x5), this.ratioY(canvas, y5));

      context.moveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2));
      context.lineTo(this.ratioX(canvas, x6), this.ratioY(canvas, y6));

      context.moveTo(this.ratioX(canvas, x7), this.ratioY(canvas, y7));
      context.quadraticCurveTo(this.ratioX(canvas, x8), this.ratioY(canvas, y8), this.ratioX(canvas, x9), this.ratioY(canvas, y9));

      // Symmetry
      context.moveTo(this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));
      context.bezierCurveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.bezierCurveTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4), this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4), this.ratioX(canvas, Constants.FaceWidth-x5), this.ratioY(canvas, y5));

      context.moveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2));
      context.lineTo(this.ratioX(canvas, Constants.FaceWidth-x6), this.ratioY(canvas, y6));

      context.moveTo(this.ratioX(canvas, Constants.FaceWidth-x7), this.ratioY(canvas, y7));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x8), this.ratioY(canvas, y8), this.ratioX(canvas, Constants.FaceWidth-x9), this.ratioY(canvas, y9));
      context.lineWidth = 3;
      context.stroke();


      context.beginPath();
      context.moveTo(this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, x10), this.ratioY(canvas, y10), this.ratioX(canvas, x11), this.ratioY(canvas, y11));

      // Symmetry
      context.moveTo(this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x10), this.ratioY(canvas, y10), this.ratioX(canvas, Constants.FaceWidth-x11), this.ratioY(canvas, y11));
      context.lineWidth = 2;
      context.stroke();


      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, x12), this.ratioY(canvas, y12), this.ratioX(canvas, x13), this.ratioY(canvas, y13));

      // Symmetry
      context.moveTo(this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x12), this.ratioY(canvas, y12), this.ratioX(canvas, Constants.FaceWidth-x13), this.ratioY(canvas, y13));
      context.lineWidth = 6;
      context.stroke();
    },
    draw_face_contour: function (canvas, context, lips_cte) {
      const x1 = 64, y1 = 352, x2 = 70, y2 = 380, x3 = 75, y3 = 492, x4 = 77, y4 = 555, x5 = 143, y5 = 634+lips_cte, x6 = 180, y6 = 685+lips_cte, x7 = Constants.FaceWidth/2, y7 = 683+lips_cte;
      const x8 = 76, y8 = 324, x9 = 80, y9 = 292, x10 = 85, y10 = 250, x11 = 110, y11 = 232, x12 = 146, y12 = 216, x13 = 180, y13 = 230, x14 = 216, y14 = 250;
      const x15 = 276, y15 = 245, x16 = 340, y16 = 230, x17 = 362, y17 = 260, x18 = 375, y18 = 284, x19 = 382, y19 = 328, x20 = 390, y20 = 348;

      context.beginPath();

      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1))
      context.quadraticCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4), this.ratioX(canvas, x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, x6), this.ratioY(canvas, y6), this.ratioX(canvas, x7), this.ratioY(canvas, y7));

      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1))
      context.quadraticCurveTo(this.ratioX(canvas, x8), this.ratioY(canvas, y8), this.ratioX(canvas, x9), this.ratioY(canvas, y9));
      context.quadraticCurveTo(this.ratioX(canvas, x10), this.ratioY(canvas, y10), this.ratioX(canvas, x11), this.ratioY(canvas, y11));
      context.quadraticCurveTo(this.ratioX(canvas, x12), this.ratioY(canvas, y12), this.ratioX(canvas, x13), this.ratioY(canvas, y13));
      context.quadraticCurveTo(this.ratioX(canvas, x14), this.ratioY(canvas, y14), this.ratioX(canvas, x15), this.ratioY(canvas, y15));
      context.quadraticCurveTo(this.ratioX(canvas, x16), this.ratioY(canvas, y16), this.ratioX(canvas, x17), this.ratioY(canvas, y17));
      context.quadraticCurveTo(this.ratioX(canvas, x18), this.ratioY(canvas, y18), this.ratioX(canvas, x19), this.ratioY(canvas, y19));
      context.quadraticCurveTo(this.ratioX(canvas, x20), this.ratioY(canvas, y20), this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));

      // Symmetry
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4), this.ratioX(canvas, Constants.FaceWidth-x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x6), this.ratioY(canvas, y6), this.ratioX(canvas, Constants.FaceWidth-x7), this.ratioY(canvas, y7));

      context.lineWidth = 3;
      context.fill();
      context.stroke();
    },
    draw_nose: function (canvas, context) {
      const x1 = 208, y1 = 420, x2 = 204, y2 = 444, x3 = 200, y3 = 456, x4 = 184, y4 = 505, x5 = 210, y5 = 507, x6 = Constants.FaceWidth/2, y6 = 525;

      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4), this.ratioX(canvas, x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, x6), this.ratioY(canvas, y6), this.ratioX(canvas, Constants.FaceWidth-x5), this.ratioY(canvas, y5));

      // Symmetry
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4), this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));

      context.stroke();
    },
    draw_dark_circles: function (canvas, context) {
      const x1 = 184, y1 = 420, x2 = 180, y2 = 426, x3 = 176, y3 = 428;
      const x4 = 164, y4 = 440, x5 = 156, y5 = 448, x6 = 150, y6 = 448;

      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x3), this.ratioY(canvas, y3));

      context.moveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4));
      context.quadraticCurveTo(this.ratioX(canvas, x5), this.ratioY(canvas, y5), this.ratioX(canvas, x6), this.ratioY(canvas, y6));
      // Symmetry
      context.moveTo(this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));

      context.moveTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x5), this.ratioY(canvas, y5), this.ratioX(canvas, Constants.FaceWidth-x6), this.ratioY(canvas, y6));

      context.lineWidth = 1;
      context.stroke();
    },
    draw_left_eye: function (canvas, context, state) {
      this.draw_eye(canvas, context, state.x1_leftEye, state.x2_leftEye, state.x3_leftEye, state.x4_leftEye, state.x5_leftEye, state.x6_leftEye, state.x7_leftEye, state.y1_leftEyeUp, state.y2_leftEyeUp, state.y3_leftEyeUp, state.y4_leftEyeUp,
        state.y5_leftEyeUp, state.y6_leftEyeUp, state.y7_leftEyeUp, state.y2_leftEyeDown, state.y3_leftEyeDown, state.y4_leftEyeDown, state.y5_leftEyeDown, state.y6_leftEyeDown,
        state.x_leftPupil, state.y_leftPupil, state.r_leftPupil, state.x_leftIris, state.y_leftIris, state.r_leftIris,
        state.x1_leftEyebrow, state.x2_leftEyebrow, state.x3_leftEyebrow, state.y1_leftEyebrowUp, state.y2_leftEyebrowUp, state.y3_leftEyebrowUp, state.y1_leftEyebrowDown, state.y2_leftEyebrowDown, state.y3_leftEyebrowDown);
      },
    draw_right_eye: function (canvas, context, state) {
      this.draw_eye(canvas, context, state.x1_rightEye, state.x2_rightEye, state.x3_rightEye, state.x4_rightEye, state.x5_rightEye, state.x6_rightEye, state.x7_rightEye, state.y1_rightEyeUp, state.y2_rightEyeUp, state.y3_rightEyeUp, state.y4_rightEyeUp,
        state.y5_rightEyeUp, state.y6_rightEyeUp, state.y7_rightEyeUp, state.y2_rightEyeDown, state.y3_rightEyeDown, state.y4_rightEyeDown, state.y5_rightEyeDown, state.y6_rightEyeDown,
        state.x_rightPupil, state.y_rightPupil, state.r_rightPupil, state.x_rightIris, state.y_rightIris, state.r_rightIris,
        state.x1_rightEyebrow, state.x2_rightEyebrow, state.x3_rightEyebrow, state.y1_rightEyebrowUp, state.y2_rightEyebrowUp, state.y3_rightEyebrowUp, state.y1_rightEyebrowDown, state.y2_rightEyebrowDown, state.y3_rightEyebrowDown);
      },
    draw_eye: function (canvas, context, x1_eye, x2_eye, x3_eye, x4_eye, x5_eye, x6_eye, x7_eye, y1_eyeUp, y2_eyeUp, y3_eyeUp, y4_eyeUp,
      y5_eyeUp, y6_eyeUp, y7_eyeUp, y2_eyeDown, y3_eyeDown, y4_eyeDown, y5_eyeDown, y6_eyeDown,
      x_pupil, y_pupil, r_pupil, x_iris, y_iris, r_iris,
      x1_eyebrow, x2_eyebrow, x3_eyebrow, y1_eyebrowUp, y2_eyebrowUp, y3_eyebrowUp, y1_eyebrowDown, y2_eyebrowDown, y3_eyebrowDown) {

      // Hole in canvas
      context.globalCompositeOperation = "destination-out"
      context.beginPath();
      this.eye_contour (canvas, context, x1_eye, x2_eye, x3_eye, x4_eye, x5_eye, x6_eye, x7_eye, y1_eyeUp, y2_eyeUp, y3_eyeUp, y4_eyeUp,
        y5_eyeUp, y6_eyeUp, y7_eyeUp, y2_eyeDown, y3_eyeDown, y4_eyeDown, y5_eyeDown, y6_eyeDown);
      context.fill();

      context.globalCompositeOperation = "destination-over"
      context.beginPath();
      // Pupil
      context.arc(this.ratioX(canvas, x_pupil), this.ratioX(canvas, y_pupil), this.ratioX(canvas, r_pupil), 0, 2 * Math.PI, false);
      context.fillStyle = "#000000";
      context.fill();

      // Iris
      context.arc(this.ratioX(canvas, x_iris), this.ratioX(canvas, y_iris), this.ratioX(canvas, r_iris), 0, 2 * Math.PI, false);
      context.fillStyle = "#1E90FF";
      context.fill();

      // Eyes_contour
      context.globalCompositeOperation = "source-over"
      context.beginPath();
      this.eye_contour (canvas, context, x1_eye, x2_eye, x3_eye, x4_eye, x5_eye, x6_eye, x7_eye, y1_eyeUp, y2_eyeUp, y3_eyeUp, y4_eyeUp,
        y5_eyeUp, y6_eyeUp, y7_eyeUp, y2_eyeDown, y3_eyeDown, y4_eyeDown, y5_eyeDown, y6_eyeDown);

      context.lineWidth = 1.5;
      context.stroke();

      // Eyebrows
      context.beginPath();
      context.moveTo(this.ratioX(canvas, x1_eyebrow), this.ratioY(canvas, y1_eyebrowUp));
      context.quadraticCurveTo(this.ratioX(canvas, x2_eyebrow), this.ratioY(canvas, y2_eyebrowUp), this.ratioX(canvas, x3_eyebrow), this.ratioY(canvas, y3_eyebrowUp));
      context.lineTo(this.ratioX(canvas, x3_eyebrow), this.ratioY(canvas, y3_eyebrowDown));
      context.quadraticCurveTo(this.ratioX(canvas, x2_eyebrow), this.ratioY(canvas, y2_eyebrowDown), this.ratioX(canvas, x1_eyebrow), this.ratioY(canvas, y1_eyebrowDown));
      context.lineWidth = 1;
      context.fillStyle = "#000000";
      context.fill();
      context.stroke();
    },
    eye_contour: function (canvas, context, x1_eye, x2_eye, x3_eye, x4_eye, x5_eye, x6_eye, x7_eye, y1_eyeUp, y2_eyeUp, y3_eyeUp, y4_eyeUp, y5_eyeUp, y6_eyeUp, y7_eyeUp,
      y2_eyeDown, y3_eyeDown, y4_eyeDown, y5_eyeDown, y6_eyeDown) {

      context.moveTo(this.ratioX(canvas, x1_eye), this.ratioY(canvas, y1_eyeUp));
      context.quadraticCurveTo(this.ratioX(canvas, x2_eye), this.ratioY(canvas, y2_eyeUp), this.ratioX(canvas, x3_eye), this.ratioY(canvas, y3_eyeUp));
      context.quadraticCurveTo(this.ratioX(canvas, x4_eye), this.ratioY(canvas, y4_eyeUp), this.ratioX(canvas, x5_eye), this.ratioY(canvas, y5_eyeUp));
      context.quadraticCurveTo(this.ratioX(canvas, x6_eye), this.ratioY(canvas, y6_eyeUp), this.ratioX(canvas, x7_eye), this.ratioY(canvas, y7_eyeUp));

      context.quadraticCurveTo(this.ratioX(canvas, x6_eye), this.ratioY(canvas, y6_eyeDown ), this.ratioX(canvas, x5_eye), this.ratioY(canvas, y5_eyeDown));
      context.quadraticCurveTo(this.ratioX(canvas, x4_eye), this.ratioY(canvas, y4_eyeDown ), this.ratioX(canvas, x3_eye), this.ratioY(canvas, y3_eyeDown));
      context.quadraticCurveTo(this.ratioX(canvas, x2_eye), this.ratioY(canvas, y2_eyeDown ), this.ratioX(canvas, x1_eye), this.ratioY(canvas, y1_eyeUp));
    },
    draw_mouth: function (canvas, context, mouthSignal, state) {
      const x1_upperUpperLip = mouthSignal.arrayMouthSignalUp[0] + 164;
      const x2_upperUpperLip = mouthSignal.arrayMouthSignalUp[1] + 178;
      const x3_upperUpperLip = mouthSignal.arrayMouthSignalUp[2] + 188;
      const x4_upperUpperLip = mouthSignal.arrayMouthSignalUp[3] + 197;
      const x5_upperUpperLip = mouthSignal.arrayMouthSignalUp[4] + 208;
      const x6_upperUpperLip = mouthSignal.arrayMouthSignalUp[5] + 218;
      const x7_upperUpperLip = mouthSignal.arrayMouthSignalUp[6] + Constants.FaceWidth/2;

      const y1_upperUpperLip = mouthSignal.arrayMouthSignalUp[7] + 570;
      const y2_upperUpperLip = mouthSignal.arrayMouthSignalUp[8] + 568;
      const y3_upperUpperLip = mouthSignal.arrayMouthSignalUp[9] + 564;
      const y4_upperUpperLip = mouthSignal.arrayMouthSignalUp[10] + 561;
      const y5_upperUpperLip = mouthSignal.arrayMouthSignalUp[11] + 562;
      const y6_upperUpperLip = mouthSignal.arrayMouthSignalUp[12] + 564;
      const y7_upperUpperLip = mouthSignal.arrayMouthSignalUp[13] + 564;


      const x1_upperLowerLip = mouthSignal.arrayMouthSignalUp[14] + 164;
      const x2_upperLowerLip = mouthSignal.arrayMouthSignalUp[15] + 178;
      const x3_upperLowerLip = mouthSignal.arrayMouthSignalUp[16] + 188;
      const x4_upperLowerLip = mouthSignal.arrayMouthSignalUp[17] + 197;
      const x5_upperLowerLip = mouthSignal.arrayMouthSignalUp[18] + 208;
      const x6_upperLowerLip = mouthSignal.arrayMouthSignalUp[19] + 218;
      const x7_upperLowerLip = mouthSignal.arrayMouthSignalUp[20] + Constants.FaceWidth/2;

      const y1_upperLowerLip = mouthSignal.arrayMouthSignalUp[21] + 570;
      const y2_upperLowerLip = mouthSignal.arrayMouthSignalUp[22] + 571;
      const y3_upperLowerLip = mouthSignal.arrayMouthSignalUp[23] + 574;
      const y4_upperLowerLip = mouthSignal.arrayMouthSignalUp[24] + 577;
      const y5_upperLowerLip = mouthSignal.arrayMouthSignalUp[25] + 578;
      const y6_upperLowerLip = mouthSignal.arrayMouthSignalUp[26] + 580;
      const y7_upperLowerLip = mouthSignal.arrayMouthSignalUp[27] + 580;

      const x1_lowerUpperLip = mouthSignal.arrayMouthSignalDown[0] + 164;
      const x2_lowerUpperLip = mouthSignal.arrayMouthSignalDown[1] + 178;
      const x3_lowerUpperLip = mouthSignal.arrayMouthSignalDown[2] + 188;
      const x4_lowerUpperLip = mouthSignal.arrayMouthSignalDown[3] + 197;
      const x5_lowerUpperLip = mouthSignal.arrayMouthSignalDown[4] + 208;
      const x6_lowerUpperLip = mouthSignal.arrayMouthSignalDown[5] + 218;
      const x7_lowerUpperLip = mouthSignal.arrayMouthSignalDown[6] + Constants.FaceWidth/2;

      const y1_lowerUpperLip = mouthSignal.arrayMouthSignalDown[7] + 570;
      const y2_lowerUpperLip = mouthSignal.arrayMouthSignalDown[8] + 571;
      const y3_lowerUpperLip = mouthSignal.arrayMouthSignalDown[9] + 574;
      const y4_lowerUpperLip = mouthSignal.arrayMouthSignalDown[10] + 577;
      const y5_lowerUpperLip = mouthSignal.arrayMouthSignalDown[11] + 578;
      const y6_lowerUpperLip = mouthSignal.arrayMouthSignalDown[12] + 580;
      const y7_lowerUpperLip = mouthSignal.arrayMouthSignalDown[13] + 580;


      const x1_lowerLowerLip = mouthSignal.arrayMouthSignalDown[14] + 164;
      const x2_lowerLowerLip = mouthSignal.arrayMouthSignalDown[15] + 178;
      const x3_lowerLowerLip = mouthSignal.arrayMouthSignalDown[16] + 188;
      const x4_lowerLowerLip = mouthSignal.arrayMouthSignalDown[17] + 197;
      const x5_lowerLowerLip = mouthSignal.arrayMouthSignalDown[18] + 208;
      const x6_lowerLowerLip = mouthSignal.arrayMouthSignalDown[19] + 218;
      const x7_lowerLowerLip = mouthSignal.arrayMouthSignalDown[20] + Constants.FaceWidth/2;

      const y1_lowerLowerLip = mouthSignal.arrayMouthSignalDown[21] + 570;
      const y2_lowerLowerLip = mouthSignal.arrayMouthSignalDown[22] + 584;
      const y3_lowerLowerLip = mouthSignal.arrayMouthSignalDown[23] + 588;
      const y4_lowerLowerLip = mouthSignal.arrayMouthSignalDown[24] + 593;
      const y5_lowerLowerLip = mouthSignal.arrayMouthSignalDown[25] + 596;
      const y6_lowerLowerLip = mouthSignal.arrayMouthSignalDown[26] + 599;
      const y7_lowerLowerLip = mouthSignal.arrayMouthSignalDown[27] + 600;


      const upperMouthCurve = getMouthCurve(7, state.upperMouthCurveA);
      const lowerMouthCurve = getMouthCurve(7, state.lowerMouthCurveA);

      // Black mouth
      context.beginPath();
      this.contour_lips_draw_left_to_right(canvas, context, x1_upperLowerLip, y1_upperLowerLip + upperMouthCurve[0], x2_upperLowerLip, y2_upperLowerLip + upperMouthCurve[1], x3_upperLowerLip, y3_upperLowerLip + upperMouthCurve[2], x4_upperLowerLip, y4_upperLowerLip + upperMouthCurve[3], x5_upperLowerLip, y5_upperLowerLip + upperMouthCurve[4], x6_upperLowerLip, y6_upperLowerLip + upperMouthCurve[5], x7_upperLowerLip, y7_upperLowerLip + upperMouthCurve[6])
      this.contour_lips_draw_right_to_left(canvas, context, x1_lowerUpperLip, y1_lowerUpperLip + lowerMouthCurve[0], x2_lowerUpperLip, y2_lowerUpperLip + lowerMouthCurve[1], x3_lowerUpperLip, y3_lowerUpperLip + lowerMouthCurve[2], x4_lowerUpperLip, y4_lowerUpperLip + lowerMouthCurve[3], x5_lowerUpperLip, y5_lowerUpperLip + lowerMouthCurve[4], x6_lowerUpperLip, y6_lowerUpperLip + lowerMouthCurve[5], x7_lowerUpperLip, y7_lowerUpperLip + lowerMouthCurve[6])
      context.fillStyle = "#000000";
      context.fill();

      // Upper lip
      context.beginPath();
      this.contour_lips_draw_left_to_right(canvas, context, x1_upperUpperLip, y1_upperUpperLip + upperMouthCurve[0], x2_upperUpperLip, y2_upperUpperLip + upperMouthCurve[1], x3_upperUpperLip, y3_upperUpperLip + upperMouthCurve[2], x4_upperUpperLip, y4_upperUpperLip + upperMouthCurve[3], x5_upperUpperLip, y5_upperUpperLip + upperMouthCurve[4], x6_upperUpperLip, y6_upperUpperLip + upperMouthCurve[5], x7_upperUpperLip, y7_upperUpperLip + upperMouthCurve[6])
      this.contour_lips_draw_right_to_left(canvas, context, x1_upperLowerLip, y1_upperLowerLip + upperMouthCurve[0], x2_upperLowerLip, y2_upperLowerLip + upperMouthCurve[1], x3_upperLowerLip, y3_upperLowerLip + upperMouthCurve[2], x4_upperLowerLip, y4_upperLowerLip + upperMouthCurve[3], x5_upperLowerLip, y5_upperLowerLip + upperMouthCurve[4], x6_upperLowerLip, y6_upperLowerLip + upperMouthCurve[5], x7_upperLowerLip, y7_upperLowerLip + upperMouthCurve[6])
      context.fillStyle = "#CD5C5C";
      context.fill();
      context.lineWidth = 2;
      context.strokeStyle = '#000000';
      context.stroke();

      // Lower lip
      context.beginPath();
      this.contour_lips_draw_left_to_right(canvas, context, x1_lowerUpperLip, y1_lowerUpperLip + lowerMouthCurve[0], x2_lowerUpperLip, y2_lowerUpperLip + lowerMouthCurve[1], x3_lowerUpperLip, y3_lowerUpperLip + lowerMouthCurve[2], x4_lowerUpperLip, y4_lowerUpperLip + lowerMouthCurve[3], x5_lowerUpperLip, y5_lowerUpperLip + lowerMouthCurve[4], x6_lowerUpperLip, y6_lowerUpperLip + lowerMouthCurve[5], x7_lowerUpperLip, y7_lowerUpperLip + lowerMouthCurve[6])
      this.contour_lips_draw_right_to_left(canvas, context, x1_lowerLowerLip, y1_lowerLowerLip + lowerMouthCurve[0], x2_lowerLowerLip, y2_lowerLowerLip + lowerMouthCurve[1], x3_lowerLowerLip, y3_lowerLowerLip + lowerMouthCurve[2], x4_lowerLowerLip, y4_lowerLowerLip + lowerMouthCurve[3], x5_lowerLowerLip, y5_lowerLowerLip + lowerMouthCurve[4], x6_lowerLowerLip, y6_lowerLowerLip + lowerMouthCurve[5], x7_lowerLowerLip, y7_lowerLowerLip + lowerMouthCurve[6])
      context.fill();
      context.stroke();
    },
    contour_lips_draw_left_to_right: function (canvas, context, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) {
      context.moveTo(this.ratioX(canvas,  x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4), this.ratioX(canvas, x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, x6), this.ratioY(canvas, y6), this.ratioX(canvas, x7), this.ratioY(canvas, y7));
      // Symmetry
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x6), this.ratioY(canvas, y6), this.ratioX(canvas, Constants.FaceWidth-x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4), this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x1), this.ratioY(canvas, y1));
    },
    contour_lips_draw_right_to_left: function (canvas, context, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) {
      context.moveTo(this.ratioX(canvas,  Constants.FaceWidth-x1), this.ratioY(canvas, y1));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x2), this.ratioY(canvas, y2), this.ratioX(canvas, Constants.FaceWidth-x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x4), this.ratioY(canvas, y4), this.ratioX(canvas, Constants.FaceWidth-x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, Constants.FaceWidth-x6), this.ratioY(canvas, y6), this.ratioX(canvas, Constants.FaceWidth-x7), this.ratioY(canvas, y7));
      // Symmetry
      context.quadraticCurveTo(this.ratioX(canvas, x6), this.ratioY(canvas, y6), this.ratioX(canvas, x5), this.ratioY(canvas, y5));
      context.quadraticCurveTo(this.ratioX(canvas, x4), this.ratioY(canvas, y4), this.ratioX(canvas, x3), this.ratioY(canvas, y3));
      context.quadraticCurveTo(this.ratioX(canvas, x2), this.ratioY(canvas, y2), this.ratioX(canvas, x1), this.ratioY(canvas, y1));
    },
    ratioX: function (canvas, x) {
      return x/465*Constants.FaceWidth;
    },
    ratioY: function (canvas, y) {
      return y/760*canvas.height;
    },
  },
  faceAnimations: {
    normal: [
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      }
    ],
    sleep: [
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 392,
          y3_leftEyeUp: 394,
          y4_leftEyeUp: 396,
          y5_leftEyeUp: 391,
          y6_leftEyeUp: 385,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 392,
          y3_rightEyeUp: 394,
          y4_rightEyeUp: 396,
          y5_rightEyeUp: 391,
          y6_rightEyeUp: 385,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      }
    ],
    blink: [
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      },
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 392,
          y3_leftEyeUp: 394,
          y4_leftEyeUp: 396,
          y5_leftEyeUp: 391,
          y6_leftEyeUp: 385,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 392,
          y3_rightEyeUp: 394,
          y4_rightEyeUp: 396,
          y5_rightEyeUp: 391,
          y6_rightEyeUp: 385,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      },
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      },
      {
        duration: 2.0,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      }
    ],
    wink_left: [
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      },
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 392,
          y3_leftEyeUp: 394,
          y4_leftEyeUp: 396,
          y5_leftEyeUp: 391,
          y6_leftEyeUp: 385,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 362,
          y2_leftEyebrowUp: 337,
          y3_leftEyebrowUp: 352,
          y1_leftEyebrowDown: 364,
          y2_leftEyebrowDown: 343,
          y3_leftEyebrowDown: 362,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      },
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      },
      {
        duration: 2.0,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      }
    ],
    wink_right: [
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      },
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 392,
          y3_rightEyeUp: 394,
          y4_rightEyeUp: 396,
          y5_rightEyeUp: 391,
          y6_rightEyeUp: 385,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 362,
          y2_rightEyebrowUp: 337,
          y3_rightEyebrowUp: 352,
          y1_rightEyebrowDown: 364,
          y2_rightEyebrowDown: 343,
          y3_rightEyebrowDown: 362,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      },
      {
        duration: 0.1,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      },
      {
        duration: 2.0,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: 0,
          lowerMouthCurveA: 0
        }
      }
    ],
    awe: [
      {
        duration: 0.25,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 366,
          y3_leftEyeUp: 362,
          y4_leftEyeUp: 359,
          y5_leftEyeUp: 366,
          y6_leftEyeUp: 371,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 394,
          y3_leftEyeDown: 396,
          y4_leftEyeDown: 398,
          y5_leftEyeDown: 393,
          y6_leftEyeDown: 387,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 366,
          y3_rightEyeUp: 362,
          y4_rightEyeUp: 359,
          y5_rightEyeUp: 366,
          y6_rightEyeUp: 371,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 394,
          y3_rightEyeDown: 396,
          y4_rightEyeDown: 398,
          y5_rightEyeDown: 393,
          y6_rightEyeDown: 387,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 140,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 310,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 316,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 140,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 310,
          y3_rightEyebrowUp: 349,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 316,
          y3_rightEyebrowDown: 359,

          upperMouthCurveA: -13,
          lowerMouthCurveA: 13
        }
      }
    ],
    skeptic: [
      {
        duration: 0.25,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 370,
          y3_leftEyeUp: 366,
          y4_leftEyeUp: 363,
          y5_leftEyeUp: 370,
          y6_leftEyeUp: 375,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 394,
          y3_leftEyeDown: 396,
          y4_leftEyeDown: 398,
          y5_leftEyeDown: 393,
          y6_leftEyeDown: 387,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 375,
          y3_rightEyeUp: 372,
          y4_rightEyeUp: 370,
          y5_rightEyeUp: 375,
          y6_rightEyeUp: 379,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 135,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 315,
          y3_leftEyebrowUp: 349,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 321,
          y3_leftEyebrowDown: 359,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 364,
          y2_rightEyebrowUp: 339,
          y3_rightEyebrowUp: 356,
          y1_rightEyebrowDown: 366,
          y2_rightEyebrowDown: 345,
          y3_rightEyebrowDown: 366,

          upperMouthCurveA: -10,
          lowerMouthCurveA: -10
        }
      }
    ],
    angry: [
      {
        duration: 0.25,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 371,
          y3_leftEyeUp: 367,
          y4_leftEyeUp: 363,
          y5_leftEyeUp: 370,
          y6_leftEyeUp: 374,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 371,
          y3_rightEyeUp: 367,
          y4_rightEyeUp: 363,
          y5_rightEyeUp: 370,
          y6_rightEyeUp: 374,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 357,
          y2_leftEyebrowUp: 332,
          y3_leftEyebrowUp: 362,
          y1_leftEyebrowDown: 359,
          y2_leftEyebrowDown: 338,
          y3_leftEyebrowDown: 372,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 357,
          y2_rightEyebrowUp: 332,
          y3_rightEyebrowUp: 362,
          y1_rightEyebrowDown: 359,
          y2_rightEyebrowDown: 338,
          y3_rightEyebrowDown: 372,

          upperMouthCurveA: -13,
          lowerMouthCurveA: -13
        }
      }
    ],
    sad: [
      {
        duration: 0.25,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 373,
          y3_leftEyeUp: 370,
          y4_leftEyeUp: 368,
          y5_leftEyeUp: 373,
          y6_leftEyeUp: 377,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 373,
          y3_rightEyeUp: 370,
          y4_rightEyeUp: 368,
          y5_rightEyeUp: 373,
          y6_rightEyeUp: 377,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 362,
          y2_leftEyebrowUp: 359,
          y3_leftEyebrowUp: 345,
          y1_leftEyebrowDown: 364,
          y2_leftEyebrowDown: 365,
          y3_leftEyebrowDown: 355,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 362,
          y2_rightEyebrowUp: 359,
          y3_rightEyebrowUp: 345,
          y1_rightEyebrowDown: 364,
          y2_rightEyebrowDown: 365,
          y3_rightEyebrowDown: 355,

          upperMouthCurveA: -13,
          lowerMouthCurveA: -13
        }
      }
    ],
    disgust: [
      {
        duration: 0.25,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 381,
          y3_leftEyeUp: 380,
          y4_leftEyeUp: 379,
          y5_leftEyeUp: 381,
          y6_leftEyeUp: 383,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 381,
          y3_rightEyeUp: 380,
          y4_rightEyeUp: 379,
          y5_rightEyeUp: 381,
          y6_rightEyeUp: 383,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 97,
          x2_leftEyebrow: 154,
          x3_leftEyebrow: 207,

          y1_leftEyebrowUp: 360,
          y2_leftEyebrowUp: 349,
          y3_leftEyebrowUp: 362,
          y1_leftEyebrowDown: 362,
          y2_leftEyebrowDown: 355,
          y3_leftEyebrowDown: 372,

          x1_rightEyebrow: Constants.FaceWidth - 97,
          x2_rightEyebrow: Constants.FaceWidth - 157,
          x3_rightEyebrow: Constants.FaceWidth - 207,

          y1_rightEyebrowUp: 360,
          y2_rightEyebrowUp: 349,
          y3_rightEyebrowUp: 362,
          y1_rightEyebrowDown: 362,
          y2_rightEyebrowDown: 355,
          y3_rightEyebrowDown: 372,

          upperMouthCurveA: -13,
          lowerMouthCurveA: -2
        }
      }
    ],
    fear: [
      {
        duration: 0.25,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 366,
          y3_leftEyeUp: 362,
          y4_leftEyeUp: 359,
          y5_leftEyeUp: 366,
          y6_leftEyeUp: 371,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 394,
          y3_leftEyeDown: 396,
          y4_leftEyeDown: 398,
          y5_leftEyeDown: 393,
          y6_leftEyeDown: 387,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 366,
          y3_rightEyeUp: 362,
          y4_rightEyeUp: 359,
          y5_rightEyeUp: 366,
          y6_rightEyeUp: 371,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 394,
          y3_rightEyeDown: 396,
          y4_rightEyeDown: 398,
          y5_rightEyeDown: 393,
          y6_rightEyeDown: 387,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 120,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 352,
          y2_leftEyebrowUp: 349,
          y3_leftEyebrowUp: 335,
          y1_leftEyebrowDown: 354,
          y2_leftEyebrowDown: 355,
          y3_leftEyebrowDown: 345,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 120,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 352,
          y2_rightEyebrowUp: 349,
          y3_rightEyebrowUp: 335,
          y1_rightEyebrowDown: 354,
          y2_rightEyebrowDown: 355,
          y3_rightEyebrowDown: 345,

          upperMouthCurveA: -13,
          lowerMouthCurveA: 8
        }
      }
    ],
    happy: [
      {
        duration: 0.25,
        state: {
          x1_leftEye: 116,
          x2_leftEye: 126,
          x3_leftEye: 142,
          x4_leftEye: 158,
          x5_leftEye: 172,
          x6_leftEye: 182,
          x7_leftEye: 188,

          y1_leftEyeUp: 384,
          y2_leftEyeUp: 371,
          y3_leftEyeUp: 368,
          y4_leftEyeUp: 366,
          y5_leftEyeUp: 371,
          y6_leftEyeUp: 375,
          y7_leftEyeUp: 384,

          y2_leftEyeDown: 392,
          y3_leftEyeDown: 394,
          y4_leftEyeDown: 396,
          y5_leftEyeDown: 391,
          y6_leftEyeDown: 385,

          x1_rightEye: Constants.FaceWidth - 116,
          x2_rightEye: Constants.FaceWidth - 126,
          x3_rightEye: Constants.FaceWidth - 142,
          x4_rightEye: Constants.FaceWidth - 158,
          x5_rightEye: Constants.FaceWidth - 172,
          x6_rightEye: Constants.FaceWidth - 182,
          x7_rightEye: Constants.FaceWidth - 188,

          y1_rightEyeUp: 384,
          y2_rightEyeUp: 371,
          y3_rightEyeUp: 368,
          y4_rightEyeUp: 366,
          y5_rightEyeUp: 371,
          y6_rightEyeUp: 375,
          y7_rightEyeUp: 384,

          y2_rightEyeDown: 392,
          y3_rightEyeDown: 394,
          y4_rightEyeDown: 396,
          y5_rightEyeDown: 391,
          y6_rightEyeDown: 385,


          x_leftPupil: 152,
          y_leftPupil: 382,
          r_leftPupil: 7,
          x_leftIris: 152,
          y_leftIris: 382,
          r_leftIris: 17,

          x_rightPupil: Constants.FaceWidth - 152,
          y_rightPupil: 382,
          r_rightPupil: 7,
          x_rightIris: Constants.FaceWidth - 152,
          y_rightIris: 382,
          r_rightIris: 17,


          x1_leftEyebrow: 92,
          x2_leftEyebrow: 140,
          x3_leftEyebrow: 200,

          y1_leftEyebrowUp: 355,
          y2_leftEyebrowUp: 324,
          y3_leftEyebrowUp: 347,
          y1_leftEyebrowDown: 357,
          y2_leftEyebrowDown: 330,
          y3_leftEyebrowDown: 357,

          x1_rightEyebrow: Constants.FaceWidth - 92,
          x2_rightEyebrow: Constants.FaceWidth - 135,
          x3_rightEyebrow: Constants.FaceWidth - 200,

          y1_rightEyebrowUp: 355,
          y2_rightEyebrowUp: 324,
          y3_rightEyebrowUp: 347,
          y1_rightEyebrowDown: 357,
          y2_rightEyebrowDown: 330,
          y3_rightEyebrowDown: 357,

          upperMouthCurveA: 10,
          lowerMouthCurveA: 10
        }
      }
    ]
  },
  mouth: {
    //arrayMouthSignalUp length = 21
    //arrayMouthSignalDown length = 24
    arrayMouthSignalUp: [16, 7, 2, 2, 2, 2, 0, 0, -4, -5, -10, -14, -18, -16, 21, 9, 4, 2, 2, 2, 0, 0, -2, -7, -13, -16, -20, -20],
    arrayMouthSignalDown: [21, 10, 4, 2, 2, 2, 0, 0, 3, 5, 11, 19, 24, 25, 16, 9, 2, 1, 2, 2, 0, 0, 4, 4, 14, 19, 23, 22],
    mouthHeight: 0,
    mouthWidth: 0
  }
}
