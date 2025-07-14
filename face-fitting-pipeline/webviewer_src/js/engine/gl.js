// eslint-disable-next-line import/no-mutable-exports
let gl;

function initGL(baseCanvas) {
  gl = baseCanvas.getContext('webgl');
  if (!gl) {
    // eslint-disable-next-line no-alert
    alert('Unable to initialize WebGL. Your browser or machine may not support it.');
    return false;
  }
  gl.extensions = {};
  gl.derivatives_supported = gl.getExtension('OES_standard_derivatives') || false;
  gl.extensions.OES_standard_derivatives = gl.derivatives_supported;
  gl.extensions.WEBGL_depth_texture = gl.getExtension('WEBGL_depth_texture') || gl.getExtension('WEBKIT_WEBGL_depth_texture') || gl.getExtension('MOZ_WEBGL_depth_texture');
  gl.extensions.OES_element_index_uint = gl.getExtension('OES_element_index_uint');
  gl.extensions.WEBGL_draw_buffers = gl.getExtension('WEBGL_draw_buffers');
  // for float textures
  gl.extensions.OES_texture_float_linear = gl.getExtension('OES_texture_float_linear');
  if (gl.extensions.OES_texture_float_linear) gl.extensions.OES_texture_float = gl.getExtension('OES_texture_float');

  gl.extensions.OES_texture_half_float_linear = gl.getExtension('OES_texture_half_float_linear');
  if (gl.extensions.OES_texture_half_float_linear) gl.extensions.OES_texture_half_float = gl.getExtension('OES_texture_half_float');

  gl.HALF_FLOAT_OES = 0x8D61;
  if (gl.extensions.OES_texture_half_float) {
    gl.HALF_FLOAT_OES = gl.extensions.OES_texture_half_float.HALF_FLOAT_OES;
  }
  return true;
}

// export default gl;
export { gl, initGL };
