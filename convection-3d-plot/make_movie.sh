# PNG encoding with ffmpeg
# Options:
#  -y         Overwrite output
#  -f image2pipe    Input format
#  -vcodec png     Input codec
#  -r $3        Frame rate
#  -i $1        Input files from cat command
#  -f mp4       Output format
#  -vcodec libx254   Output codec
#  -pix_fmt yuv420p  Output pixel format
#  -preset slower   Prefer slower encoding / better results
#  -crf 20       Constant rate factor (lower for better quality)
#  -vf "scale..."   Round to even size
#  $2         Output file
#!/bin/bash

function png2mp4(){
  cat $1* | ffmpeg \
    -y \
    -f image2pipe \
    -vcodec png \
    -framerate $3 \
    -i - \
    -f mp4 \
    -vcodec libx264 \
    -pix_fmt yuv420p \
	-preset slower \
    -crf 20 \
    -vf "scale=trunc(in_w/2)*2:trunc(in_h/2)*2" \
    $2
}

png2mp4 profiles_mu_1em03_R_4ep02_Ro_1ep00_Nx_128_Ny_128_Nz_512/write_ profiles_mu_1em03_R_4ep02_Ro_1ep00_Nx_128_Ny_128_Nz_512.mp4 10 
#png2mp4 frames_profiles_mu_1em3_Nx_128_Nz_512_longer/write_ convection_2d_d2_profiles_mu_1em3_Nx_128_Nz_512_longer.mp4 10
