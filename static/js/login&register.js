const container = document.getElementById('container');
const registerBtn = document.getElementById('register');
const loginBtn = document.getElementById('login');

registerBtn.addEventListener('click', () => {
    container.classList.add("active");
});

loginBtn.addEventListener('click', () => {
    container.classList.remove("active");
});

document.addEventListener('DOMContentLoaded', function () {
    const formType = document.getElementById('form-type').innerText;
    if (formType == 'register') {
            container.classList.add('active');
    }
    else {
            container.classList.remove('active');
    }
});