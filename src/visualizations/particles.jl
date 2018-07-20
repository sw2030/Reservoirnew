# Installation:
# Pkg.checkout.(
#     ("Makie", "GLVisualize", "GLAbstraction", "GLFW"),
#     ("sd/newdesign", "sd/atlas", "sd/fixvao", "sd/compat")
# )
using Makie, FileIO, GeometryTypes, Colors
cd(@__DIR__)
psps = load("pressure3.jld")["pressure_ver3"]
mini, maxi = mapreduce(extrema, (a, b)-> (min(a[1], b[1]), max(a[2], b[2])), psps)
psps_n = map(psps) do vol
    Float32.((vol .- mini) ./ (maxi - mini))
end

volume100 = last(psps_n)
scene = Scene()
g_area = map(scene.px_area) do a
    IRect(0, 0, widths(a)[1], 50)
end
vol_area = map(scene.px_area) do a
    w, h = widths(a)
    IRect(0, 50, w - 120, h - 50)
end
gui = Scene(scene, g_area)
campixel!(gui)

scene3d = Scene(scene, vol_area)
cam3d!(scene3d)

r = linspace(-1, 1, size(volume100, 1))
marker_scale = 0.025f0
markersize = vec(volume100 .* marker_scale)
dims = size(volume100)
grid = (Point3f0.(ind2sub.((dims,), 1:prod(dims))) .- 1f0) ./ maximum(dims)
cmap = reverse(attribute_convert(:Spectral, key"colormap"()))
most_abundant_value = median(median.(psps_n))
x = length(cmap) * most_abundant_value
cmap = RGBAf0.(color.(cmap), 0.3)
# cmap[floor(Int, x)] = RGBAf0(color(cmap[floor(Int, x)]), 0.0)
# cmap[ceil(Int, x)] = RGBAf0(color(cmap[ceil(Int, x)]), 0.0)
msize = Vec2f0.((markersize) .* marker_scale)
particles = scatter!(
    scene3d, grid,
    markersize = msize,
    colormap = cmap,
    colorrange = Vec2f0(0.0, 1.0),
    marker_offset = Vec2f0(0.0),
    intensity = vec(markersize),
)

N = length(psps_n)
t = slider(translated(gui), 1:length(psps_n); sliderlength = 150, sliderheight = 40)
foreach(t[:value]) do idx
    if checkbounds(Bool, psps_n, idx)
        markersize .= vec(psps_n[idx])
        broadcast!(msize, markersize, marker_scale) do s, ms
            Vec2f0(s * ms)
        end
        particles[:intensity][] = markersize
        particles[:markersize][] = msize
    end
end
b = playbutton(translated(gui), 1:length(psps_n)) do idx
    move!(t, idx)
end
vbox(b, t)
colorlegend(scene, :Spectral, (mini, maxi))
scene
# record(scene, "pressure_particle.mp4", 1:200) do idx
#     yield()
# end
