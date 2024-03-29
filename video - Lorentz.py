import os, numpy as np, matplotlib.pyplot as plt, cv2
import wave, struct, moviepy.editor as mpy
from mpl_toolkits.mplot3d import Axes3D

audio_data = # audio data, see `get_audio`.
picture_0_data = 

class Lorentz_system:
    def __init__(self, R=28, sig=10, beta=8/3):
        self.cons = [R, sig, beta]
        self.point = list()
    def equation(self, p):
        R, sig, beta = self.cons
        x, y, z = p
        return np.array((sig*(y-x), x*(R-z)-y, x*y-beta*z))
    def rk4(self, p, dt):
        k1 = self.equation(p)
        k2 = self.equation(p + k1*dt/2)
        k3 = self.equation(p + k2*dt/2)
        k4 = self.equation(p + k3*dt)
        return (k1 + 2*k2 + 2*k3 + k4) / 6
    def run(self, dt, total_time):
        p = self.point[-1]
        for _ in np.arange(0, total_time, dt):
            p = p + self.rk4(p, dt)*dt
            self.point.append(p)
    def get_coordinates(self): return np.array(self.point).transpose(1, 0)


def get_audio(data):
    leng = len(data) // 2
    audio = np.zeros(leng).astype(np.uint)
    for i in range(leng): audio[i] = int(data[2*i: 2*i+2], 16)
    with open('./audio.mp3', 'wb') as wf: wf.write(bytes(audio.tolist()))

def get_pic(data):
    leng = len(data) // 2
    picdata = np.zeros(leng).astype(np.int)
    for i in range(leng): picdata[i] = int(data[2*i: 2*i+2], 16)
    picdata.shape = (960, 1280, 3)
    return picdata

def composite_video():
    readed_frame = [0]
    def read_frame(t):
        frame = cv2.imread('./frames/{}.jpg'.format(readed_frame[0]))
        readed_frame[0] = readed_frame[0] + 1
        return frame[:, :, ::-1]
    clip = mpy.VideoClip(read_frame, duration=7511/30)
    clip.write_videofile('./no_audio.mp4', fps=30)
    os.system('ffmpeg -i "./no_audio.mp4" -i "./audio.mp3" -shortest "Finall.mp4"')


if __name__ == '__main__':
    if not os.path.isdir('./frames'): os.mkdir('./frames')
    print('Getting audio...')
    get_audio(audio_data)
    del audio_data
    print('Getting picture...')
    pic = get_pic(picture_0_data)
    del picture_0_data
    Lo_sys = Lorentz_system()
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    zmin, zmax = 0, 0

    for i_frame in range(7512):
        print('Rendering frames... ({}/7513)'.format(i_frame+1), end='\r')
        if i_frame<45: render = pic
        elif i_frame==45:
            render = pic
            del pic
        else:

            rho = 100*(i_frame-45) / 7467
            Lo_sys.cons[0] = rho
            Lo_sys.point = [np.array((.1, .1, .1))]
            Lo_sys.run(.001, 30)
            x, y, z = Lo_sys.get_coordinates()
            xmin = min(xmin, x.min()); xmax = max(xmax, x.max())
            ymin = min(ymin, y.min()); ymax = max(ymax, y.max())
            zmin = min(zmin, z.min()); zmax = max(zmax, z.max())

            plt.close('all')
            fig = plt.figure(1)
            ax = Axes3D(fig)
            plt.title('rho = {:.3f}'.format(rho))
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
            ax.scatter(.1, .1, .1, color='black', lw=2)
            ax.plot(x, y, z, color='orange', lw=0.75)
            plt.savefig('./frames/cache.jpg', dpi=200)
            render = 255 - cv2.imread('./frames/cache.jpg', 1)

        cv2.imwrite('./frames/{}.jpg'.format(i_frame), render)
    composite_video()
