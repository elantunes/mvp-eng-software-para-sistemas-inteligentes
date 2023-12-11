window.onclick = function(event) {
    if (event.target.classList.contains('modal'))
        event.target.style.display = 'none';
}

function closeModal(modal) {
    modal.style.display = 'none';
}

function load() {
    const closes = document.querySelectorAll('.close');
    
    closes.forEach( el => {
        el.addEventListener('click', function() {
            el.closest('.modal').style.display='none';
        });
    });
}

document.addEventListener("DOMContentLoaded", load, false);