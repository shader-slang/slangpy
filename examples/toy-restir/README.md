## Example: Toy renderer with ReSTIR

This example uses ReSTIR [1,2,3] to render the view of an animated 2D toy scene. Artificial noise is introduced
to the path contributions to model path tracing noise. Samples are reused between pixels and frames with
ReSTIR for lower-noise integration of the path tracing noise and the pixel footprint.

The sample reuse between pixels and frames is based on generalized resampled importance sampling (GRIS) [2]
with the generalized balance heuristic; this is thoroughly documented in the "Gentle Introduction to ReSTIR"
SIGGRAPH course [3]. Temporal reuse is made fully unbiased by using the prior-frame target
function in the multiple importance sampling (MIS) in resampling. Temporal reuse follows integer
motion vectors, which is not ideal, but works fine for low-frequency content [4]. The implementation is
theoretically proper, but lacks improvements like multiple spatial neighbors or G-buffer based sample
rejection in spatial reuse, which could make reuse much more robust in practice. A proper application
would further apply a high-quality denoiser to the ReSTIR output.

For context and additional information, please see the ReSTIR course "A Gentle Introduction to ReSTIR:
Path Reuse in Real-Time" (Wyman et al. 2023, [3]), https://intro-to-restir.cwyman.org/, especially the course notes.

This application outputs the ReSTIR frames to 'tev'; first 1spp frames and then ReSTIR frames.

![Toy ReSTIR](toy-restir.png)

## References

[1] **Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting**  
    _Benedikt Bitterli, Chris Wyman, Matt Pharr, Peter Shirley, Aaron Lefohn, Wojciech Jarosz_  
    ACM Transactions on Graphics (TOG), Volume 39, Issue 4, August 2020  
    https://doi.org/10.1145/3386569.3392481

[2] "Generalized resampled importance sampling: Foundations of ReSTIR"  
    Daqi Lin, Markus Kettunen, Benedikt Bitterli, Jacopo Pantaleoni, Cem Yuksel, Chris Wyman  
    ACM Transactions on Graphics (TOG), Volume 41, Issue 4, July 2022  
    https://doi.org/10.1145/3528223.3530158
    
[3] "A gentle introduction to ReSTIR: Path reuse in real-time"  
    Chris Wyman, Markus Kettunen, Daqi Lin, Benedikt Bitterli, Cem Yuksel, Wojciech Jarosz, Pawel Kozlowski, Giovanni De Francesco  
    SIGGRAPH '23: ACM SIGGRAPH 2023 Courses, July 2023  
    https://intro-to-restir.cwyman.org/

[4] "Area ReSTIR: Resampling for real-time defocus and antialiasing"  
    Song Zhang, Daqi Lin, Markus Kettunen, Cem Yuksel, Chris Wyman  
    ACM Transactions on Graphics (TOG), Volume 43, Issue 4, July 2024  
    https://doi.org/10.1145/3658210

