import skvideo.io
import numpy as np
import argparse
import tqdm

from RobustPCA import RPCA, RPCA_inexact


def main(args):
    input_path = args.i
    output_path = args.o

    # Read video
    video = skvideo.io.vread(input_path)

    # Parameters
    l = args.l
    rho = args.r
    
    if args.all:
        print("Optimizing using all frames")

        # Normalize
        video = video / 255

        h, w, c = video.shape[1:]
        video_flatten = video.reshape((video.shape[0], -1))

        # Robust PCA
        mu = 1.25 / np.linalg.norm(video_flatten, ord=2, axis=(0,1))
        A, E = RPCA_inexact(video_flatten, l, mu, rho)

        # Reconstruct
        clean_video = []
        noise = []
        for i in range(len(video_flatten)):
            clean_video.append((A[i].reshape((h, w, c))*255).astype(np.uint8))
            noise.append((E[i].reshape((h, w, c))*255).astype(np.uint8))

        # Save video
        skvideo.io.vwrite(output_path, np.array(clean_video))

        # Save noise
        if args.save_noise:
            skvideo.io.vwrite(output_path.replace('.mp4', '_noise.mp4'), np.array(noise))



    else:
        print("Optimizing frame by frame")

        clean_video = []
        noise = []
        for i in tqdm.tqdm(range(len(video))):
            # Normalize
            frame = video[i] / 255

            # Robust PCA
            mu = 1.25 / np.linalg.norm(frame[:,:,0], ord=2, axis=(0,1))
            RA, RE = RPCA_inexact(frame[:,:,0], l, mu, rho)

            mu = 1.25 / np.linalg.norm(frame[:,:,1], ord=2, axis=(0,1))
            GA, GE = RPCA_inexact(frame[:,:,1], l, mu, rho)

            mu = 1.25 / np.linalg.norm(frame[:,:,2], ord=2, axis=(0,1))
            BA, BE = RPCA_inexact(frame[:,:,2], l, mu, rho)

            # Reconstruct
            A = (np.stack((RA, GA, BA), axis=2)*255).astype(np.uint8)
            E = (np.stack((RE, GE, BE), axis=2)*255).astype(np.uint8)

            clean_video.append(A)
            noise.append(E)

        # Save video
        skvideo.io.vwrite(output_path, clean_video)

        # Save noise
        if args.save_noise:
            skvideo.io.vwrite(output_path.replace('.mp4', '_noise.mp4'), noise)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Part1: Video Denoising')
    parser.add_argument('--i', '--input', type=str, required=True, help='input video path')
    parser.add_argument('--o','--output', type=str, required=True, help='output video path')
    parser.add_argument('--l', '--lambda', type=float, default=0.1, help='lambda')
    parser.add_argument('--r', '--rho', type=float, default=1.5, help='rho')
    parser.add_argument('--save_noise', action='store_true', help='Save noise video')
    parser.add_argument('--all', action='store_true', help='Optimizing using all frames')
    main(parser.parse_args())