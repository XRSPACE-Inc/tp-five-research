import { setFrameDirty, screenSize, clamp } from './utils.js';
// eslint-disable-next-line import/no-mutable-exports
let input;

function onMouseWheel(event) {
  setFrameDirty();
  const e = window.event || event; // old IE support
  input.wheelDelta = -clamp((e.wheelDelta || -e.detail), -1, 1);

  if (input.eventListeners.length > 0) {
    for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
      if (input.eventListeners[i].OnMouseWheel !== undefined) {
        input.eventListeners[i].OnMouseWheel(input.wheelDelta);
      }
    }
  }
}

function onMouseDown(event) {
  setFrameDirty();
  const e = window.event || event; // old IE support
  if (e.touches !== undefined) {
    e.preventDefault();
  }
  input.mouseDown = true;
  if (e.touches !== undefined) {
    input.lastMouseX = e.touches[0].clientX * devicePixelRatio;
    input.lastMouseY = e.touches[0].clientY * devicePixelRatio;
  } else {
    input.lastMouseX = e.clientX;
    input.lastMouseY = e.clientY;
  }
  const touchEvent = {
    x: input.lastMouseX,
    y: screenSize[1] - input.lastMouseY,
    type: e.buttons,
  };

  if (input.eventListeners.length > 0) {
    for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
      if (input.eventListeners[i].OnTouchStart !== undefined) {
        input.anyObjectTouched = input.eventListeners[i].OnTouchStart(touchEvent);
        if (input.anyObjectTouched) break;
      }
    }
  }

  return true;
}

function onMouseUp(e) {
  setFrameDirty();
  input.mouseDown = false;
  const touchEvent = {
    x: input.lastMouseX,
    y: screenSize[1] - input.lastMouseY,
    type: e.buttons,
  };

  if (input.eventListeners.length > 0) {
    for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
      if (input.eventListeners[i].OnTouchEnd !== undefined) {
        input.eventListeners[i].OnTouchEnd(touchEvent);
      }
    }
  }
  input.anyObjectTouched = false;
  return true;
}

function onMouseMove(e) {
  if (!input.mouseDown) {
    input.checkHover(e);
    return false;
  }
  setFrameDirty();

  const newX = e.clientX;
  const newY = e.clientY;

  const deltaX = newX - input.lastMouseX;
  const deltaY = newY - input.lastMouseY;
  const touchEvent = {
    x: newX,
    y: screenSize[1] - newY,
    type: e.buttons,
    deltaX,
    deltaY,
  };

  if (input.eventListeners.length > 0) {
    for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
      if (input.eventListeners[i].OnTouch !== undefined) {
        input.eventListeners[i].OnTouch(touchEvent);
      }
    }
  }

  input.lastMouseX = newX;
  input.lastMouseY = newY;
  return true;
}

function onMouseEnterEvent(e) {
  input.mouseDown = input.mouseDownOut !== 0 && input.mouseDownOut === e.buttons;
  input.anyObjectTouched = input.mouseDown && input.anyObjectTouchedOut;

  input.keys.Alt = e.altKey ? 1 : 0;
  input.keys.Control = e.ctrlKey ? 1 : 0;
  input.keys.Shift = e.shiftKey ? 1 : 0;

  if (input.eventListeners.length > 0) {
    for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
      if (input.eventListeners[i].OnMouseEnterWindow !== undefined) {
        input.eventListeners[i].OnMouseEnterWindow();
      }
    }
  }
}

function onMouseOutEvent(e) {
  input.mouseDownOut = e.buttons;
  input.anyObjectTouchedOut = input.anyObjectTouched;
  input.mouseDown = false;
  input.anyObjectTouched = false;

  if (input.onMouseOut !== undefined) input.onMouseOut();

  if (input.eventListeners.length > 0) {
    for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
      if (input.eventListeners[i].OnMouseOutWindow !== undefined) {
        input.eventListeners[i].OnMouseOutWindow();
      }
    }
  }
}

function onTouchMove(e) {
  setFrameDirty();
  if (e.touches) {
    e.preventDefault();
  }
  if (!input.mouseDown) {
    input.checkHover(e);
    return false;
  }

  const newX = e.touches[0].clientX * devicePixelRatio;
  const newY = e.touches[0].clientY * devicePixelRatio;
  let deltaX = newX - input.lastMouseX;
  let deltaY = newY - input.lastMouseY;
  if (deltaX === undefined || Number.isNaN(deltaX)) deltaX = 0;
  if (deltaY === undefined || Number.isNaN(deltaY)) deltaY = 0;
  deltaX *= 0.4;
  deltaY *= 0.4;
  let touchEvent;
  if (e.touches.length > 1) {
    const touch2X = e.touches[1].clientX * devicePixelRatio;
    const touch2Y = e.touches[1].clientY * devicePixelRatio;
    touchEvent = {
      type: 1,
      x: newX,
      y: screenSize[1] - newY,
      deltaX,
      deltaY,
      x2: touch2X,
      y2: screenSize[1] - touch2Y,
    };
  } else {
    touchEvent = {
      type: 1,
      x: newX,
      y: screenSize[1] - newY,
      deltaX,
      deltaY,
    };
  }

  if (input.eventListeners.length > 0) {
    for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
      if (input.eventListeners[i].OnTouch !== undefined) {
        input.eventListeners[i].OnTouch(touchEvent);
      }
    }
  }

  input.lastMouseX = newX;
  input.lastMouseY = newY;

  return true;
}

function onDragFile(e) {
  setFrameDirty();
  input.mouseDown = false;
  input.anyObjectTouched = false;
  e.preventDefault();
  e.stopPropagation();
  // console.log(e);

  const dt = e.dataTransfer;
  const { files } = dt;
  const { x, y } = e;
  const pos = [x, screenSize[1] - y];

  if (files.length > 0) {
    if (input.onFilesEnter !== undefined) {
      input.onFilesEnter(files, pos);
    }
  }
}

function onKeyDown(e) {
  setFrameDirty();
  if (!input.keys[e.key]) {
    // console.log(`keydown ${e.key}`);
    input.keys[e.key] = 1;
    if (input.eventListeners.length > 0) {
      for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
        if (input.eventListeners[i].OnKeyDown) {
          input.eventListeners[i].OnKeyDown(e);
        }
      }
    }
  } else {
    // console.log(`keypressing ${e.key}`);
    if (input.eventListeners.length > 0) {
      for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
        if (input.eventListeners[i].OnKeyPress) {
          if (this.warningFirstTime) {
            this.warningFirstTime = true;
            // eslint-disable-next-line no-console
            console.warn('input manager warning: key pressing event only fire for last key.');
          }
          input.eventListeners[i].OnKeyPress(e);
        }
      }
    }
    input.keys[e.key] = 1;
  }
}

function onKeyUp(e) {
  setFrameDirty();

  // console.log(`keyup ${e.key}`);
  input.keys[e.key] = 0;
  if (input.eventListeners.length > 0) {
    for (let i = input.eventListeners.length - 1; i >= 0; i -= 1) {
      if (input.eventListeners[i].OnKeyUp) {
        input.eventListeners[i].OnKeyUp(e);
      }
    }
  }
}

function preventDefaultEvents(e) {
  e.preventDefault();
  e.stopPropagation();
}

class InputManager {
  constructor() {
    this.container = document.getElementById('container');

    this.container.addEventListener('touchstart', onMouseDown, false);
    this.container.addEventListener('touchend', onMouseUp, false);
    this.container.addEventListener('touchcancel', onMouseUp, false);
    this.container.addEventListener('touchmove', onTouchMove, false);

    this.container.addEventListener('mouseenter', onMouseEnterEvent, false);
    this.container.addEventListener('mouseout', onMouseOutEvent, false);

    this.container.addEventListener('mousewheel', onMouseWheel, false); // others
    this.container.addEventListener('DOMMouseScroll', onMouseWheel, false); // firefox

    this.container.addEventListener('dragenter', preventDefaultEvents, false);
    this.container.addEventListener('dragover', preventDefaultEvents, false);
    this.container.addEventListener('dragleave', preventDefaultEvents, false);
    this.container.addEventListener('drop', onDragFile, false);

    this.container.addEventListener('dblclick', preventDefaultEvents, false);

    document.addEventListener('keydown', onKeyDown, false);
    document.addEventListener('keyup', onKeyUp, false);
    // document.addEventListener('keypress', onKeyPress, false);

    this.container.onmousedown = onMouseDown;
    this.container.onmouseup = onMouseUp;
    this.container.onmousemove = onMouseMove;

    this.wheelDelta = 0.0;

    this.mouseDown = false;
    this.mouseDownOut = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;

    this.onMouseOut = undefined;

    this.onFilesEnter = undefined;

    this.lockTarget = undefined;
    this.anyObjectTouched = false;
    this.anyObjectTouchedOut = false;
    this.eventListeners = [];

    this.keys = {};
    this.warningFirstTime = true;
  }

  checkHover(e) {
    let newX;
    let newY;
    if (e.touches) {
      newX = e.touches[0].clientX * devicePixelRatio;
      newY = e.touches[0].clientY * devicePixelRatio;
    } else {
      newX = e.clientX;
      newY = e.clientY;
    }
    const touchEvent = {
      type: 1,
      x: newX,
      y: screenSize[1] - newY,
    };
    for (let i = this.eventListeners.length - 1; i >= 0; i -= 1) {
      if (input.eventListeners[i].OnMouse !== undefined) {
        input.eventListeners[i].OnMouse(touchEvent);
      }
    }
  }
}

input = new InputManager();

// eslint-disable-next-line import/prefer-default-export
export { input };
