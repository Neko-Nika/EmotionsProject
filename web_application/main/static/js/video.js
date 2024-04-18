window.onload = () => {
    processBtn.onclick = processVideo
}

async function processVideo(event) {
    document.getElementById("error").hidden = true
    document.getElementById("error").innerHTML = ""

    video_file = document.getElementById("inputVideo").files[0]
    if (video_file == null || video_file == ""){
        document.getElementById("error").hidden = false
        document.getElementById("error").innerHTML = "Обязательно загрузите видео для обработки"
        return
    }

    const formData = new FormData();
    formData.append("video", video_file)
    
    var url = '/api/upload_video'

    let response = await fetch(url, {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        data = await response.json()
        if (data['success'] == true) {
            ;
        } else {
            console.log(data['message'])
            document.getElementById("error").hidden = false
            document.getElementById("error").innerHTML = data['message']
        }
    } 
}
