window.onload = () => {
    processBtn.onclick = processVideo
    processBtn.hidden = false

    downloadBtn.hidden = true
    downloadBtn.disabled = true
    downloadBtn.href = false
    downloadBtn.download = false

    spinner.hidden = false
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
    processBtn.hidden = true
    downloadBtn.hidden = false

    let response = await fetch(url, {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        data = await response.json()
        if (data['success'] == true) {
            spinner.hidden = true
            processBtn.hidden = true
            downloadBtn.disabled = false
            downloadBtn.hidden = false
            
            downloadBtn.href = "http://127.0.0.1:8000/media/uploaded_videos/" + data['output_filename']
            downloadBtn.download = data['output_filename']
            downloadBtn.textContent = 'Скачать'
        } else {
            console.log(data['message'])
            document.getElementById("error").hidden = false
            document.getElementById("error").innerHTML = data['message']
        }
    } 
}
