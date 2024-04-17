window.onload = () => {
    signInBtn.onclick = loginAccount
}

async function loginAccount(event) {
    if (!login_account.reportValidity()){
        console.log("Error in form")
        return
      }

    let response = await fetch('/api/login_account', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        
        body: JSON.stringify({
            'email': emailInput.value,
            'password': passwordInput.value
        })
    });

    if (response.ok) {
        data = await response.json()
        if (data['success'] == true) {
            window.location.replace("/");
        } else {
            console.log(data['message'])
            document.getElementById("error").hidden = false
            document.getElementById("error").innerHTML = data['message']
        }
    } 
}