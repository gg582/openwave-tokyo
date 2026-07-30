#version 330 core
// Background pass: fullscreen triangle, no vertex attributes needed.
out vec2 vNdc;
void main()
{
    const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    vec2 pos = p[gl_VertexID];
    vNdc = pos;
    gl_Position = vec4(pos, 0.999, 1.0);   // far plane
}
