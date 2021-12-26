'''运行这份代码你需要:
python3, numpy, pillow, moviepy, (may be ffmpeg?)
直至2020-06-21, 依赖库都为最新版,
made by nyasyamorina.'''
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import editor as mpy

from numpy import pi, floor, clip as clamp, arange as range
E = 1e-6        # 误差, 用来控制zeta函数计算时的精度(或运行速度)






# 在画幅上的曲线是 zeta(real0+i*imag0)->zeta(real1+i*imag1)
real0 = 0.5 ; real1 = 0.5
imag0 = 0.  ; imag1 = 35.

fig_size = (2160,3840)      # {int} 画幅大小, 反正4k也只是吃多点内存, 没啥区别
ori      = (1080,1362)      # {int|float} (origin) 画幅里的坐标原点
ulen     = 528              # {int|float} (unit length) 画幅里的单位长度

fps  = 60                   # {int} 帧率还要解释吗
z_ps = 1/3                  # {float} (z per second) 这个代表在输入平面上前进的速度, 比如说real0=imag0=0, real1=imag1=3, dz_ps=2 那么整条视频长度为 1.5*sqrt2
p_pf = 1                    # {int} (points per frame) 这个是渲染每一帧里选取点的数量, 越小曲线越光滑, 不过就我测试, 60fps这个值1就够了 (写来只是玩玩)
vcol = 0.2                  # {float} (velocity of color) 色轮的转动速度, 颜色选取是使用hsv颜色空间, vcol=1.就代表一秒转动一圈

# 还有更多小的参数在下面代码里面






def zeta( s ):                  # zeta_function_from_Havil_2003
    '''
    |             1        ∞     1       n              n
    |  ζ(s) = --------- *  Σ   ------- *  Σ   (-1)^k * (   ) * (k+1)^(-s)
    |         1-2^(1-s)   n=0  2^(n+1)   k=0             k
    '''
    res = 0+0j

    n = -1
    while True:
        n += 1

        step = 0+0j

        comb = 1
        for k in range(n+1):

            step += comb * (k+1) ** -s
            comb *= (k-n) / (k+1)

        step *= 2 **- (n+1)
        res += step

        if abs(step) < E: break
    return res / ( 1 - 2 ** (1-s) )


def _zeta_without_complex( r,i ):
    ''' 用来给想要在C或其他没有复数的语言写的例子, 
        效果与上面的一模一样, 只不过输入输出变成两个浮点数, 代表实部和虚部 '''
    R = I = 0.

    n = -1
    while True:
        n += 1

        a = b = 0.

        comb = 1
        for k in range(n+1):

            A = comb * (k+1) **- r
            t = -np.log(k+1) * i
            a += A * np.cos(t)
            b += A * np.sin(t)
            comb *= (k-n) / (k+1)

        A = 2 **- (n+1)
        a *= A
        b *= A
        R += a
        I += b

        if np.sqrt(a*a+b*b) < E: break

    A = 2 ** (1-r)
    t = -np.log(2) * i
    a = 1- A * np.cos(t)
    b =    A * np.sin(t)
    d = a*a + b*b
    return (R*a - I*b) / d,   (R*b + I*a) / d


def get_RGB( h,s,v ):       # 一个从hsv颜色空间返回RGB颜色代码
    h %= 2*pi
    # h in [0,2pi), s&v in [0,1]
    ang = 3 * h / pi
    hi = floor(ang) % 6
    p = v * (1-s)
    q = v * (1- (ang-hi) * s)
    t = v * (1- (1+hi-ang) * s)

    rgb = (v,t,p) if hi==0 else((q,v,p) if hi==1 else((p,v,t) if hi==2 else(
          (p,q,v) if hi==3 else((t,p,v) if hi==4 else((v,p,q) )))))
    return '#{:02X}{:02X}{:02X}'.format(*( int(255.*c) for c in rgb ))







if 1:                       # 这里用了很长的篇幅生成坐标系图片, 完全可以 收起skip
    minecraft_font = { 'format': (('font_height','font_width'), (('p0y','p0x'), ('p1y','p1x'), ...)), 'spacing': 1,
        # 这个mc字体并没有把全部字符收录进去, 但是用来生成坐标系是够了的
        '0': ((7,5), ((0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,3),(2,4),(3,0),(3,2),(3,4),(4,0),(4,1),(4,4),(5,0),(5,4),(6,1),(6,2),(6,3))),
        '1': ((7,5), ((0,2),(1,1),(1,2),(2,2),(3,2),(4,2),(5,2),(6,0),(6,1),(6,2),(6,3),(6,4))),
        '2': ((7,5), ((0,1),(0,2),(0,3),(1,0),(1,4),(2,4),(3,2),(3,3),(4,1),(5,0),(6,0),(6,1),(6,2),(6,3),(6,4))),
        '3': ((7,5), ((0,1),(0,2),(0,3),(1,0),(1,4),(2,4),(3,2),(3,3),(4,4),(5,0),(5,4),(6,1),(6,2),(6,3))),
        '4': ((7,5), ((0,3),(0,4),(1,2),(1,4),(2,1),(2,4),(3,0),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4),(5,4),(6,4))),
        '5': ((7,5), ((0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(2,0),(2,1),(2,2),(2,3),(3,4),(4,4),(5,0),(5,4),(6,1),(6,2),(6,3))),
        '6': ((7,5), ((0,2),(0,3),(1,1),(2,0),(3,0),(3,1),(3,2),(3,3),(4,0),(4,4),(5,0),(5,4),(6,1),(6,2),(6,3))),
        '7': ((7,5), ((0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,4),(2,4),(3,3),(4,2),(5,2),(6,2))),
        '8': ((7,5), ((0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,4),(3,1),(3,2),(3,3),(4,0),(4,4),(5,0),(5,4),(6,1),(6,2),(6,3))),
        '9': ((7,5), ((0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,4),(3,1),(3,2),(3,3),(3,4),(4,4),(5,3),(6,1),(6,2))),
        '-': ((7,5), ((3,0),(3,1),(3,2),(3,3),(3,4))),
        'i': ((7,1), ((0,0),(2,0),(3,0),(4,0),(5,0),(6,0))), }

    # 啊, 在这个程序写完之后才发现有更好的方法生成坐标系图片, 但是已经懒得改了
    # 用这种方法生成的坐标系图片在scale比较大时不能很好地对上坐标点准确的位置, 但是scale太小时又会造成线条太细看不清
    scale = 3.      # 设置这个放大倍数是为了避免坐标系线条过细, 就是在低分辨生成坐标系再使用PIL放大, 繁琐但是有效, 反正只运行一次
    p_size    = 3           # 坐标点比坐标轴突出的长度
    p_density = 1           # 整数坐标点之间的小坐标点的密度, 0就是不要小坐标点
    len_pn    = 3           # 坐标点与数字之间的间隙, 我真的不会起名字了, qiao
    scale_ori = ( int(ori[0]/scale), int(ori[1]/scale) )
    scale_ulen = ulen / scale

    crood = np.zeros(  (int(fig_size[0]/scale), int(fig_size[1]/scale))  ,dtype=np.uint8)
    crood [scale_ori[0], :] = 1
    crood [:, scale_ori[1]] = 1


    def put_number( string, lefttop ):      # 这个方法用来在crood上面摆mc字体, 当然只是用来写整数坐标的数字而已
        global crood                        # 小数坐标的...太复杂了, 可以自己试着改一下这段方法和下面的循环
        y,x = lefttop
        for m in string:
            for a,b in minecraft_font[m][1]:
                try: crood [y+a, x+b] = 1
                except IndexError: continue
            x += minecraft_font[m][0][1] + minecraft_font['spacing']
    put_number( '0' ,( scale_ori[0]+p_size+len_pn+1, scale_ori[1]-p_size-minecraft_font['0'][0][1]-len_pn ))

    crooding = 0.                               # 下面这堆就是添加坐标点的定型文而已, 没什么好看的
    while True:         # Re+
        crooding += 1 / (p_density+1)
        croodX = int(0.5+  scale_ulen * crooding + scale_ori[1]  )      # 下一个坐标点在画幅的位置
        if croodX >= crood.shape[1]: break                              # 如果超过屏幕范围就没必要显示啦
        crood [scale_ori[0]-p_size:scale_ori[0]+p_size+1, croodX] = 1   # 画上坐标点
        near_int = int(crooding+0.49)                                   # 如果是整数的坐标点就写上数字, 否则跳过
        if not abs( crooding - near_int ) < 1e-4: continue              # 鬼知道会不会有神秘的浮点数误差出现
        string = str(near_int)
        strlen = (len(string)-1)*minecraft_font['spacing'] + sum([ minecraft_font[m][0][1] for m in string ])
        put_number( string ,( scale_ori[0]+p_size+len_pn+1, croodX-(strlen-1)//2 ))     # 写上数字
    crooding = 0.
    while True:         # Re-
        crooding -= 1 / (p_density+1)
        croodX = int(0.5+  scale_ulen * crooding + scale_ori[1]  )
        if croodX < 0: break
        crood [scale_ori[0]-p_size:scale_ori[0]+p_size+1, croodX] = 1
        near_int = int(crooding-0.49)
        if not abs( crooding - near_int ) < 1e-4: continue
        string = str(near_int)
        strlen = (len(string)-1)*minecraft_font['spacing'] + sum([ minecraft_font[m][0][1] for m in string ])
        put_number( string ,( scale_ori[0]+p_size+len_pn+1, croodX-(strlen-1)//2 ))
    crooding = 0.
    while True:         # Im+
        crooding += 1 / (p_density+1)
        croodY = int(0.5-  scale_ulen * crooding + scale_ori[0]  )
        if croodY < 0: break
        crood [croodY, scale_ori[1]-p_size:scale_ori[1]+p_size+1] = 1
        near_int = int(crooding+0.49)
        if not abs( crooding - near_int ) < 1e-4: continue
        string = str(near_int)+'i'
        strlen = (len(string)-1)*minecraft_font['spacing'] + sum([ minecraft_font[m][0][1] for m in string ])
        put_number( string ,( croodY-3, scale_ori[1]-strlen-p_size-len_pn ))
    crooding = 0.
    while True:         # Im-
        crooding -= 1 / (p_density+1)
        croodY = int(0.5-  scale_ulen * crooding + scale_ori[0]  )
        if croodY >= crood.shape[0]: break
        crood [croodY, scale_ori[1]-p_size:scale_ori[1]+p_size+1] = 1
        near_int = int(crooding-0.49)
        if not abs( crooding - near_int ) < 1e-4: continue
        string = str(near_int)+'i'
        strlen = (len(string)-1)*minecraft_font['spacing'] + sum([ minecraft_font[m][0][1] for m in string ])
        put_number( string ,( croodY-3, scale_ori[1]-strlen-p_size-len_pn ))

crood = Image.fromarray(255*crood).resize(fig_size[::-1],Image.NEAREST)







curve  = Image.new( 'RGB', fig_size[::-1] )
line_draw = ImageDraw.Draw( curve )

z0 = real0 +1j* imag0       # 输入平面的起点
z1 = real1 +1j* imag1       # 输入平面的终点, 不过计算时并不会用到这个值, 只能说"希望"最后一帧刚好等于这个值
dz = (z1-z0) / abs(z1-z0) * z_ps / fps / p_pf       # 输入平面里的步长
dc = 2*pi                 * vcol / fps / p_pf       # 与上一个对应的色环变化值

z = z0
c = 0.
zeta_z = zeta( z )
endX = ori[1] + ulen * zeta_z.real      # 线段的终点位置, 一开始的终点当然是起点啦
endY = ori[0] - ulen * zeta_z.imag







lw        = 5               # {int} 线段宽度

end_point = True            # 每帧都在曲线尾部加一个小圆点
ep_radius = 7.              # 小圆点的半径
ep_color  = None            # 小圆点的颜色代码, None就使用曲线最后一点的颜色

end_str   = True            # 在视频里加一个显示当前位置的字符串
lefttop   = None            # 这个字符串的位置, None就跟随ep
font_file = 'consola.ttf'   # 字体文件, PIL文档说会在系统字体文件夹里面找字体的
font_size = 50              # 字体大小
ft_color  = None            # 字体颜色, 跟ep_color一样
try: font = ImageFont.truetype( font_file, font_size )
except BaseException as error:
    print( 'Warning: font load error:\n{}: {}'.format(type(error),error) )
    font = ImageFont.load_default()







# 下面一大段注释掉的 和 get_frame里的 是生成投稿在b站上面那条视频的,
# 把注释去掉就可以生成一模一样的视频了, 当然前提是准备好需要的bgm和字体 (或者删掉)

#jimaku = {  'format': (('start_time', 'duration', 'fade_in_time', 'fade_out_time'), ('positionX', 'positionY')),
#    '你现在看到的是zeta函数的一部分'                                 :((  5.0,3.0, 0.5,0.5), (2025,2011)),
#    'yap, 是著名的黎曼猜想里提到是zeta函数'                          :((  9.0,3.0, 0.5,0.5), (2025,2011)),
#    '17年看到3b1b的视频时, 就已经被zeta函数的美震住了'               :(( 20.0,3.0, 0.5,0.5), (2025,2011)),
#    '当时就已经有想做这条视频的念头'                                 :(( 24.0,3.0, 0.5,0.5), (2025,2011)),
#    '从那时电脑都没开过几次一直到现在小会python'                     :(( 28.0,3.0, 0.5,0.5), (2025,2011)),
#    '最近终于回想起当时的遗愿, 于是就有了这条视频'                   :(( 32.0,3.0, 0.5,0.5), (2025,2011)),
#    'by the way, 生成这条视频的源码可以在简介里下载到'               :(( 35.0,3.0, 0.5,0.5), (2025,2011)),
#    '源码的内容不止这条视频'                                         :(( 39.0,3.0, 0.5,0.5), (2025,2011)),
#    '里面有很多的参数可以自己设置来玩玩'                             :(( 43.0,3.0, 0.5,0.5), (2025,2011)),
#    '也希望里面的备注可以帮到有需要的人'                             :(( 47.0,3.0, 0.5,0.5), (2025,2011)),
#    '最后说一句:   黎曼, 永远滴神'                                   :(( 50.0,3.0, 0.5,0.5), (2025,2011)),
#    'Hope you enjoy this video'                                      :(( 55.0,3.0, 0.5,0.5), (2025,2011))   }
#jimaku_font  = ImageFont.truetype( 'Alibaba-PuHuiTi-Medium.ttf' ,70)
#hope_font    = ImageFont.truetype( 'Raven Script DEMO.ttf'      ,130)
#nihonji_font = ImageFont.truetype( 'BIZ-UDGothicB.ttc'          ,65)






def get_frame(t):                       # 渲染一帧的函数
    global curve, z, c, endX, endY

    for _ in range( p_pf ):
        startX = endX; startY = endY
        z += dz; c += dc

        zeta_z = zeta( z )
        endX = ori[1] + ulen * zeta_z.real
        endY = ori[0] - ulen * zeta_z.imag

        line_draw.line( [(startX,startY), (endX,endY)], width=lw, fill=get_RGB(c,1,1) )

    frame = curve.copy()                      # curve只是用来画曲线, 而其他东西需要后期加上
    frame_draw = ImageDraw.Draw( frame )
    end_color = get_RGB(c,1,1)

    if end_point:
        frame_draw.ellipse( 
                [(endX-ep_radius,endY-ep_radius), (endX+ep_radius,endY+ep_radius)],
                fill = ep_color if ep_color is not None else end_color )

    if end_str:
        frame_draw.text(
                lefttop if lefttop is not None else (endX,endY),
                'zeta({:.2f})'.format( z ).replace('j','i')  ,font=font,
                fill = ft_color if ft_color is not None else end_color )

    #for string, info in jimaku.items():
    #    if string == 'format': continue
    #    start = info[0][0]; end = start + info[0][1]
    #    if t >= start and t < end:
    #        color = int( 255* min(1., (t-start)/info[0][2], (end-t)/info[0][3] ))
    #        color = '#{:02X}{:02X}{:02X}'.format(*(  color  for _ in range(3)))
    #        frame_draw.text(
    #                info[1],  string  ,fill=color,
    #                font=jimaku_font if not string.startswith('Hope') else hope_font )
    #if t >= 180.0 and t < 190.0:
    #    color = int( 255* min(1., (t-180.)/0.75, (190.-t)/0.75 ))
    #    frame_draw.text(
    #        (2025,2011), 'bgm: 鍵盤は躍る - 魔法使いの夜', font=nihonji_font,
    #        fill='#{:02X}{:02X}{:02X}'.format(*(color for _ in range(3))))

    frame_draw.bitmap((0,0),  crood  ,fill='#FFFFFF')
    return np.asarray( frame )



#mpy.VideoClip(get_frame,duration=  208.5  ).set_audio(mpy.AudioFileClip( '鍵盤は躍る.mp3' )).write_videofile( 'render_out.mp4', fps=fps,bitrate='50000k')
mpy.VideoClip(get_frame,duration=  abs(z1-z0) / z_ps  -1/fps).write_videofile( 'render_out.mp4', fps=fps,bitrate='50000k')
