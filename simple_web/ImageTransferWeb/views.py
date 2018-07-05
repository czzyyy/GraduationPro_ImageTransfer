# -*- coding: utf-8 -*-
import os
import time
import json
import threading

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from django.core.mail import EmailMessage
from ImageTransferWeb.models import DIYTask

from django.views.decorators.csrf import csrf_exempt
# from sys import path
# path.append(r'F:\python_code\GraduationPro\ImageTransferWeb\my_models\fast_art')
from ImageTransferWeb.my_models.fast_art.transfer_eval import evaluate
from ImageTransferWeb.my_models.fast_real.transfer_eval_real import evaluate_real
from ImageTransferWeb.algorithms.slow_art.slow_neural_style import start_slow_neural_style
from ImageTransferWeb.algorithms.slow_real.slow_real_style import start_slow_real_style

# Create your views here.


@csrf_exempt
def home(request):
    # get img nums and names from show,except for the three kept imgs.
    file_dir = os.path.join(settings.BASE_DIR, 'ImageTransferWeb/static/image/show/').replace('\\', '/')
    files = None
    for root, dirs, files in os.walk(file_dir):
        pass
    context = dict()
    context['show_imgs'] = ["/static/image/show/" + item for item in files if
                            (item != 'eightchicago.png' and item != 'fivechicago.png' and item != 'onechicago.png')]
    return render(request, 'Home.html', context)


@csrf_exempt
def upload_online(request):
    if request.method == 'POST':
        print(str(request.FILES['file']))
        handle_uploaded_file(request.FILES['file'], str(request.FILES['file']), 'online')
        status = 0
        return HttpResponse(json.dumps({
            "status": status
        }))
    return HttpResponse("Failed")


@csrf_exempt
def upload_share(request):
    if request.method == 'POST':
        print(str(request.FILES['file']))
        handle_uploaded_file(request.FILES['file'], str(request.FILES['file']), 'share')
        status = 0
        return HttpResponse(json.dumps({
            "status": status
        }))
    return HttpResponse("Failed")


@csrf_exempt
def upload_diy(request):
    if request.method == 'POST':
        print(str(request.FILES['file']))
        handle_uploaded_file(request.FILES['file'], str(request.FILES['file']), 'diy')
        status = 0
        return HttpResponse(json.dumps({
            "status": status
        }))
    return HttpResponse("Failed")


@csrf_exempt
def handle_uploaded_file(file, filename, type):
    static_root_path = None
    if type == 'online':
        static_root_path = os.path.join(settings.BASE_DIR, 'ImageTransferWeb/static/image/user/').replace('\\', '/')
    elif type == 'share':
        static_root_path = os.path.join(settings.BASE_DIR, 'ImageTransferWeb/static/image/show/').replace('\\', '/')
    elif type == 'diy':
        static_root_path = os.path.join(settings.BASE_DIR, 'ImageTransferWeb/static/image/diy/').replace('\\', '/')

    if not os.path.exists(static_root_path):
        os.mkdir(static_root_path)

    with open(static_root_path + filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)


@csrf_exempt
def save_diy_info(request):
    # 获取数据 姓名 邮箱 两个图的地址 保存在数据库中 并设置一个 flag看有没有发送
    if request.method == 'POST':
        new_name = str(request.POST.get("name", "111"))
        new_email = str(request.POST.get("email", "222"))
        new_diy_user = str(request.POST.get("diy_user", "333"))
        new_diy_style = str(request.POST.get("diy_style", "444"))
        new_diy_style_detail = str(request.POST.get("diy_style_detail", 555))

        # 保存在数据库中,执行线程任务
        new_task = DIYTask()
        new_task.name = new_name
        new_task.email = new_email
        new_task.diy_user = new_diy_user
        new_task.diy_style = new_diy_style
        new_task.diy_style_detail = new_diy_style_detail
        new_task.save()

        print(new_name + new_email + new_diy_user + new_diy_style + new_diy_style_detail)

        status = 0
        result = new_name

        prints = PrintThread()
        prints.start()

        return HttpResponse(json.dumps({
            "status": status,
            "result": result
        }))
    return HttpResponse("Failed")


# for test
class PrintThread(threading.Thread):
    def run(self):
        print("start.... %s" % (self.getName(),))
        # 查找数据库,进行算法
        static_root_path = os.path.join(settings.BASE_DIR, 'ImageTransferWeb').replace('\\', '/')
        tasks = DIYTask.objects.all()
        for t in tasks:
            # 如果没有处理过就处理
            if not t.flag:
                # 先是设为处理过
                t.flag = True
                t.save()
                # run algorithm
                style_path = static_root_path + t.diy_style
                content_path = static_root_path + t.diy_user
                the_type = str(t.diy_style_detail)
                result = static_root_path + t.diy_style
                try:
                    if the_type == '1':
                        # art
                        art_save_path = static_root_path + '/static/output/slow_art/'
                        result = start_slow_neural_style(style_path=style_path, content_path=content_path,
                                                         save_path=art_save_path)
                        print('art over: ', result)
                    else:
                        # real
                        real_save_path = static_root_path + '/static/output/slow_real/'
                        real_seg_path = static_root_path + '/static/output/slow_real/segmentation/'
                        result = start_slow_real_style(style_path=style_path, content_path=content_path,
                                                       save_path=real_save_path, segmentation_path=real_seg_path)
                        print('real over: ', result)
                except FileNotFoundError:
                    print('FileNotFoundError')
                    t.flag = False
                    t.save()
                except ValueError:
                    print('ValueError')
                    t.flag = False
                    t.save()
                except IOError:
                    print('IOError')
                    t.flag = False
                    t.save()
                except RuntimeError:
                    print('RuntimeError')
                    t.flag = False
                    t.save()

                # result = static_root_path + t.diy_style

                # 发送邮件
                from_email = settings.DEFAULT_FROM_EMAIL
                content = t.name + "您好，本邮件是图片风格转换结果，请查收~"
                msg = EmailMessage(subject='图像风格转化邮件', body=content, from_email=from_email, to=[t.email])
                msg.content_subtype = 'html'
                msg.encoding = 'utf-8'
                # 添加附件（可选）
                # 方法一
                fp = open(result, 'rb')
                msg_image = fp.read()
                fp.close()
                msg.attach('result.jpg', msg_image)
                # 发送
                res = msg.send()

                print(t.email + "  return:" + str(res))

                # 若没成功，改回来
                if res is not 1:
                    t.flag = False
                    t.save()
        print("end.... %s" % (self.getName(),))


@csrf_exempt
def transfer_style(request):
    if request.method == 'POST':
        print(request.POST)
        get_style = str(request.POST.get("style", "111"))
        get_img = str(request.POST.get("img", "222"))
        static_root_path = os.path.join(settings.BASE_DIR, 'ImageTransferWeb').replace('\\', '/')
        # style_path = static_root_path + get_style
        content_path = static_root_path + get_img
        # 处理图片
        save_path = None
        if 'style/art' in get_style:
            output_save_path = '/static/output/fast_art/' + get_img.split('/')[-1]
            if 'landscape-2.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/eight/save/transfer_net_one.ckpt'
            elif 'the_shipwreck_of_the_minotaur.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/four/save/transfer_net_one.ckpt'
            elif 'The_Starry_Night.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/one/save/transfer_net_one.ckpt'
            elif 'wave.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/two/save/transfer_net_one.ckpt'
            elif 'semi-nude-with-colored-skirt-and-raised-arms-1911.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/five/save/transfer_net_one.ckpt'
            elif 'young-woman-with-a-bouquet-of-roses.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/six/save/transfer_net_one.ckpt'
            elif 'quintais-de-lisboa-1956.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/nine/save/transfer_net_one.ckpt'
            elif 'erin-hanson-cedar-breaks-color-50x70.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/seven/save/transfer_net_one.ckpt'
            elif 'impression-sunrise.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/ten/save/transfer_net_one.ckpt'
            elif 'rain_princess.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/three/save/transfer_net_one.ckpt'
            elif 'untitled-5.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/eleven/save/transfer_net_one.ckpt'
            elif 'the_scream.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_art/stored_models/twelve/save/transfer_net_one.ckpt'
            evaluate(model_path=save_path, test_image_path=content_path, image_save_path=static_root_path + output_save_path)
        else:
            output_save_path = '/static/output/fast_real/' + get_img.split('/')[-1]
            if 'bright_indoor.png' in get_style:
                save_path = static_root_path + '/my_models/fast_real/stored_models/three/save/transfer_net_two.ckpt'
            elif 'road_house.png' in get_style:
                save_path = static_root_path + '/my_models/fast_real/stored_models/one/save/transfer_net_two.ckpt'
            elif 'light_street.png' in get_style:
                save_path = static_root_path + '/my_models/fast_real/stored_models/two/save/transfer_net_two.ckpt'
            elif 'sky.jpg' in get_style:
                save_path = static_root_path + '/my_models/fast_real/stored_models/four/save/transfer_net_two.ckpt'
            evaluate_real(model_path=save_path, test_image_path=content_path, image_save_path=static_root_path + output_save_path)
        print('ok')

        status = 0
        result = output_save_path
        return HttpResponse(json.dumps({
            "status": status,
            "result": result
        }))
    return HttpResponse("Failed")