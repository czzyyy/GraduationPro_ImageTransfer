// Avoid `console` errors in browsers that lack a console.
(function() {
    var method;
    var noop = function () {};
    var methods = [
        'assert', 'clear', 'count', 'debug', 'dir', 'dirxml', 'error',
        'exception', 'group', 'groupCollapsed', 'groupEnd', 'info', 'log',
        'markTimeline', 'profile', 'profileEnd', 'table', 'time', 'timeEnd',
        'timeline', 'timelineEnd', 'timeStamp', 'trace', 'warn'
    ];
    var length = methods.length;
    var console = (window.console = window.console || {});

    while (length--) {
        method = methods[length];

        // Only stub undefined methods.
        if (!console[method]) {
            console[method] = noop;
        }
    }
}());

// Place any jQuery/helper plugins in here.
$(document).ready(function(){

    $('.owl-carousel').owlCarousel({
        loop:true,
        margin:30,
        responsiveClass:true,
        responsive:{
            0:{
                items:1,
                nav:true
            },
            600:{
                items:1,
                nav:false
            },
            1000:{
                items:3,
                nav:true,
                loop:false
            }
        }
    })

});

//czy
$(document).ready(function() {
    $(".choose").click(function() {
        var choose = $(this).attr("id");
        alert(choose);
        $(".choose-img").attr("src","/static/image/style/"+choose);
        $(".choose-img").attr("id",choose);
        $(".online-box-styles").toggle();
        $(".online-box-show").toggle();
    });
    $(".success-back").click(function() {
         window.location.reload();
         window.location.href="#2";
         $(".online-box-content").toggle();
         $(".online-box-styles").toggle();
         $(".online-box-success").toggle();
         $("#my-pro").css("width", "4%");
    });
    $("#show-back").click(function(){
        $(".online-box-styles").toggle();
        $(".online-box-show").toggle();
    });
});

//手风琴
$(document).ready(function() {
	var Accordion = function(el, multiple) {
		this.el = el || {};
		this.multiple = multiple || false;

		// Variables privadas
		var links = this.el.find('.link');
		// Evento
		links.on('click', {el: this.el, multiple: this.multiple}, this.dropdown)
	}

	Accordion.prototype.dropdown = function(e) {
		var $el = e.data.el;
			$this = $(this),
			$next = $this.next();

		$next.slideToggle();
		$this.parent().toggleClass('open');

		if (!e.data.multiple) {
			$el.find('.submenu').not($next).slideUp().parent().removeClass('open');
		};
	}

	var accordion = new Accordion($('#accordion'), false);
});

//上传文件
OpenFile = function(flag){
    if(flag==0){
         document.getElementById("btn-file").click();
    }
    else if(flag==1){
        if($(".email-form").is(":hidden") && $(".style-detail").is(":hidden")){
            document.getElementById("btn-diy-user-file").click();
        }
    }
    else if(flag==2){
        if($(".email-form").is(":hidden") && $(".style-detail").is(":hidden")){
            document.getElementById("btn-diy-style-file").click();
        }
    }
}

//ajax上传文件代码
$(document).ready(function() {
   var uploading = false;
   var filename;
   $("#btn-file").on("cancel", function(){
        uploading = false
   });
   $("#btn-file").on("change", function(){
       if(uploading){
           alert("文件正在上传中，请稍候...");
           return false;
       }
       fileName = $(this).val().split('\\');
       var formData = new FormData();
       formData.append('file', $('#btn-file')[0].files[0]);
       $.ajax({
           url: '/ImageTransfer/upload_online/',
           type: 'POST',
           cache: false,
           data: formData,
           processData: false,
           contentType: false,
           beforeSend: function(){
               uploading = true;
           },
           success : function(data) {
               data = JSON.parse(data);
               if(data["status"] == 0){
                    alert(fileName[fileName.length-1]);
                    $("#user-img").attr("src", "/static/image/user/" + fileName[fileName.length-1]);
               }
               uploading = false;
           }
       });
   });

   $("#btn-diy-user-file").on("cancel", function(){
        uploading = false
   });
   $("#btn-diy-user-file").on("change", function(){
       if(uploading){
           alert("文件正在上传中，请稍候...");
           return false;
       }
       fileName = $(this).val().split('\\');
       var formData = new FormData();
       formData.append('file', $('#btn-diy-user-file')[0].files[0]);
       $.ajax({
           url: '/ImageTransfer/upload_diy/',
           type: 'POST',
           cache: false,
           data: formData,
           processData: false,
           contentType: false,
           beforeSend: function(){
               uploading = true;
           },
           success : function(data) {
               data = JSON.parse(data);
               if(data["status"] == 0){
                    alert(fileName[fileName.length-1]);
                    $("#diy-user-img").attr("src", "/static/image/diy/" + fileName[fileName.length-1]);
               }
               uploading = false;
           }
       });
   });

   $("#btn-diy-style-file").on("cancel", function(){
        uploading = false
   });
   $("#btn-diy-style-file").on("change", function(){
       if(uploading){
           alert("文件正在上传中，请稍候...");
           return false;
       }
       fileName = $(this).val().split('\\');
       var formData = new FormData();
       formData.append('file', $('#btn-diy-style-file')[0].files[0]);
       $.ajax({
           url: '/ImageTransfer/upload_diy/',
           type: 'POST',
           cache: false,
           data: formData,
           processData: false,
           contentType: false,
           beforeSend: function(){
               uploading = true;
           },
           success : function(data) {
               data = JSON.parse(data);
               if(data["status"] == 0){
                    alert(fileName[fileName.length-1]);
                    $("#diy-style-img").attr("src", "/static/image/diy/" + fileName[fileName.length-1]);
                    $(".style-detail").toggle("normal");
               }
               uploading = false;
           }
       });
   });
});

//email
OpenEmail = function(){
    var diy_user_img = $("#diy-user-img").attr("src");
    var diy_style_img = $("#diy-style-img").attr("src");
    if(diy_user_img == "/static/image/web/default.png" || diy_style_img == "/static/image/web/default.png"){
        alert("DIY两张图片都要选择哦");
        return;
    }
    if($(".email-form").is(":hidden") && $(".style-detail").is(":hidden")){
        $(".email-form").toggle("normal");
    }
}
CancelEmail = function(){
    if($(".email-form").is(":visible")){
        $(".email-form").toggle("normal");
    }
}
SendEmail = function(){
        var f=document.getElementById("contact");
        var name = f.name.value;
        var email = f.email.value;
        var diy_user = $("#diy-user-img").attr("src");
        var diy_style = $("#diy-style-img").attr("src");
        var diy_style_detail = $('input[name="optionsRadios"]:checked').val();
        var postdata = {
                "name":name,
                "email":email,
                "diy_user":diy_user,
                "diy_style":diy_style,
                "diy_style_detail":diy_style_detail,
        };
        alert(name + email +diy_user + diy_style + diy_style_detail);
        //不加 这里async: false, socket 会报错
        $.ajax({
           url: '/ImageTransfer/save_diy_info/',
           type: 'POST',
           data: postdata,
           async: false,
           beforeSend: function(){
               uploading = true;
           },
           success : function(data) {
               data = JSON.parse(data);
               if(data["status"] == 0){
                   alert(data["result"] + "提交成功，算法结束会发送邮件");
                   window.location.reload();
                   window.location.href="#3";
               }
               uploading = false;
           }
       });
}

StyleDetail = function(){
    $(".style-detail").toggle("normal");
}

//风格转化
//进度条http://blog.csdn.net/u013897685/article/details/73653064
StartTransfer = function(){
    //判断一下用户是否选择了要风格化的图片 和 风格，可以预先放置一个默认的图片，检查是不是变化了
    var user_img = $("#user-img").attr("src");
    if(user_img == "/static/image/web/default.png"){
        alert("未选择图片");
        $(".online-box-styles").toggle();
        $(".online-box-show").toggle();
    }
    else{
        var choose_style = $(".choose-img").attr("src");
        var p = 4;
        var stop = 0;
        var result;
        alert("风格" + choose_style + "图片" + user_img);
        window.setTimeout("go_progress()",1000);
            var postdata = {
                "style":choose_style,
                "img":user_img,
            };
            $.ajax({
                url: '/ImageTransfer/transfer_style/',
                type: 'POST',
                data: postdata,
                beforeSend: function(){

                },
                success : function(data) {
                     stop = 1;
                     data = JSON.parse(data);
                     //alert(data["status"] + data["result"]);
                     if (data["status"] == 0) {
                          result = data["result"];
//                          $("#success-img").attr("src", data["result"]);
//                          $(".online-box-styles").hide();
//                          $(".online-box-show").hide();
//                          $(".online-box-content").hide();
//                          $(".online-box-success").toggle();
                     } else {
                         alert("转换失败");
                     }
                }
            });
        change_ui = function(){
            $("#success-img").attr("src", result);
            $(".online-box-styles").hide();
            $(".online-box-show").hide();
            $(".online-box-content").hide();
            $(".online-box-success").toggle();
        }
        go_progress = function(){
            p += 1; //进度条每秒走
            if(stop==1){
                $("#my-pro").css("width","100%");
                window.setTimeout("change_ui()",1000);
                return;
            }
            $("#my-pro").css("width", p + "%");
            window.setTimeout("go_progress()",1000);
        }
    }
}

//sleep = function(d){
// var start = new Date().getTime();
// while(new Date().getTime() < start + d);
//}

//UploadShare分享上传
UploadShare = function(){
   document.getElementById("btn-upload").click();
}
//ajax上传文件代码
$(document).ready(function() {
   var uploading = false;
   var filename;
   $("#btn-upload").on("cancel", function(){
        uploading = false
   });
   $("#btn-upload").on("change", function(){
       if(uploading){
           alert("文件正在上传中，请稍候...");
           return false;
       }
       fileName = $(this).val().split('\\');
       var formData = new FormData();
       formData.append('file', $('#btn-upload')[0].files[0]);
       $.ajax({
           url: '/ImageTransfer/upload_share/',
           type: 'POST',
           cache: false,
           data: formData,
           processData: false,
           contentType: false,
           beforeSend: function(){
               uploading = true;
           },
           success : function(data) {
               data = JSON.parse(data);
               if(data["status"] == 0){
                   alert("上传成功");
                  //重新请求 ImageTransfer/home/
                  window.location.reload();
                  window.location.href="#4";
               }
               uploading = false;
           }
       });
   });
});

//DownloadShare分享下载 http://www.zhangxinxu.com/wordpress/2017/07/js-text-string-download-as-html-json-file/
//http://www.zhangxinxu.com/wordpress/2016/04/know-about-html-download-attribute/
//base64
DownloadShare = function(flag){
    var domImg;
    if(flag==0){
        domImg = document.getElementsByClassName("dg-center")[0].getElementsByTagName("img")[0];
    }
    else if(flag==1){
        domImg = document.getElementById("success-img");
    }
    alert(domImg.src);
    var eleLink = document.createElement('a');
        eleLink.download = 'image_transfer.jpg';
        eleLink.style.display = 'none';
        // 图片转base64地址
        var canvas = document.createElement('canvas');
        var context = canvas.getContext('2d');
        canvas.width = domImg.naturalWidth;
        canvas.height = domImg.naturalHeight;
        context.drawImage(domImg, 0, 0);
        eleLink.href = canvas.toDataURL('image/*');
        // 触发点击
        document.body.appendChild(eleLink);
        eleLink.click();
        // 然后移除
        document.body.removeChild(eleLink);
}