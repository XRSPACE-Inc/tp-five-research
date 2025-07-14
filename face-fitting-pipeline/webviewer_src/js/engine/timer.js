// timer will be update at app.frame
export default class Timer {
  constructor() {
    this.time = 0.0;
    this.deltaTime = 0.0;
    this.frame = 0;
  }
}

const timer = new Timer();

export { timer };
