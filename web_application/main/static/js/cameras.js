window.onload = () => {
    createBtn.onclick = addNewCamera
}

async function addNewCamera(event) {
    if (!camera_creation.reportValidity()){
        console.log("Error in form")
        return
      }

    let response = await fetch('/api/check_connection_available', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        
        body: JSON.stringify({
            'link': linkInput.value
        })
    });

    let connected = false
    if (response.ok) {
        data = await response.json()
        if (data['success'] == true) {
            connected = true
        } else {
            console.log(data['message'])
            closeModalBtn.click()
            document.getElementById("error").hidden = false
            document.getElementById("error").innerHTML = data['message']
        }
    }

    if (!connected) {
        return;
    }

    response = await fetch('/api/add_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        
        body: JSON.stringify({
            'name': nameInput.value,
            'link': linkInput.value
        })
    });

    if (response.ok) {
        data = await response.json()
        if (data['success'] == true) {
            window.location.reload()
        } else {
            console.log(data['message'])
            closeModalBtn.click()
            document.getElementById("error").hidden = false
            document.getElementById("error").innerHTML = data['message']
        }
    }

}

async function deleteCamera(btn) {
    var id = btn.getAttribute("camera_id")
    let response = await fetch('/api/delete_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        
        body: JSON.stringify({
            'camera_id': id
        })
    });

    if (response.ok) {
        data = await response.json()
        if (data['success'] == true) {
            window.location.reload()
        } else {
            console.log(data['message'])
            closeModalBtn.click()
            document.getElementById("error").hidden = false
            document.getElementById("error").innerHTML = data['message']
        }
    }
}
