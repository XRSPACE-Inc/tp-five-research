/* eslint-disable no-console */
import { isMobile } from './utils.js';

const LogType = Object.freeze({ Log: 0, Warning: 1, Error: 2 });

export default class Logger {
  constructor() {
    this.canvas = document.getElementById('text');
    this.ctx = this.canvas.getContext('2d');
    if (isMobile) this.font = '1.2em "Fira Sans", serif';
    else this.font = '0.8em "Fira Sans", serif';

    this.canvas.width = 800;
    this.canvas.height = 800;

    this.logs = [];
    // show stack not including logger.
    this.showStack = false;
    // shows full stack including logger.
    this.showFullStack = false;

    const self = this;

    this.oldLog = console.log;
    // stacked log
    console.slog = (...args) => {
      if (self.showFullStack) {
        // eslint-disable-next-line no-param-reassign
        try { throw new Error(); } catch (e) { args[0] += `${e.stack.replace('Error\n', '\n')}`; }
      } else if (self.showStack) {
        // eslint-disable-next-line no-param-reassign
        try { throw new Error(); } catch (e) { args[0] += `\n${e.stack.split('\n').splice(2).join('\n')}`; }
      }

      self.setLog(LogType.Log, args[0]);

      self.oldLog.apply(console, args);
    };

    this.oldWarn = console.warn;
    // stacked warn
    console.swarn = (...args) => {
      if (self.showFullStack) {
        // eslint-disable-next-line no-param-reassign
        try { throw new Error(); } catch (e) { args[0] += `${e.stack.replace('Error\n', '\n')}`; }
      } else if (self.showStack) {
        // eslint-disable-next-line no-param-reassign
        try { throw new Error(); } catch (e) { args[0] += `\n${e.stack.split('\n').splice(2).join('\n')}`; }
      }

      self.setLog(LogType.Warning, args[0]);

      self.oldWarn.apply(console, args);
    };

    this.oldError = console.error;
    // stacked error
    console.serror = (...args) => {
      if (self.showFullStack) {
        // eslint-disable-next-line no-param-reassign
        try { throw new Error(); } catch (e) { args[0] += `${e.stack.replace('Error\n', '\n')}`; }
      } else if (self.showStack) {
        // eslint-disable-next-line no-param-reassign
        try { throw new Error(); } catch (e) { args[0] += `\n${e.stack.split('\n').splice(2).join('\n')}`; }
      }

      self.setLog(LogType.Error, args[0]);

      self.oldError.apply(console, args);
    };

    this.logIndex = 0;
    this.logMax = 30;
    this.enabled = true;
  }

  Resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.drawLogs();
  }

  setLog(logType, message) {
    this.logs[this.logIndex] = { logType, message };
    this.logIndex += 1;
    if (this.logIndex === this.logMax) this.logIndex = 0;
    this.drawLogs();
  }

  drawLogs() {
    if (!this.enabled) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    let c;
    let max;
    let i;
    let imax;
    let messages;
    let line = 0;
    for (c = 0, max = this.logs.length; c < max; c += 1) {
      const { logType, message } = this.logs[c];
      switch (logType) {
        case LogType.Warning:
          this.ctx.fillStyle = 'yellow';
          break;
        case LogType.Error:
          this.ctx.fillStyle = 'red';
          break;
        default:
          this.ctx.fillStyle = 'white';
          break;
      }
      this.ctx.font = this.font;
      messages = message.split('\n');
      for (i = 0, imax = messages.length; i < imax; i += 1) {
        this.ctx.fillText(messages[i], 10, 32 + line * 18);
        line += 1;
      }
    }
  }

  setEnable(enable) {
    if (this.enabled !== enable) {
      this.enabled = enable;
      if (!this.enabled) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      }
    }
  }
}

const logger = new Logger();
// export default Logger;
export { logger };
