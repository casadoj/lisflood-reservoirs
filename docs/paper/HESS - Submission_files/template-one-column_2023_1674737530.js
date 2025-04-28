document.addEventListener("DOMContentLoaded", function(event) {
    const banner = document.getElementById('banner');
    if (typeof banner !== 'undefined' && banner !== null) {
        const header = document.querySelector('header.d-print-none');
        if (typeof header !== 'undefined' && header !== null) {
            header.setAttribute('id', 'has-banner');
        }
    }

    const homeHeader = document.getElementById('headers-content-container');
    if (typeof homeHeader !== 'undefined' && homeHeader !== null) {
        const headerH1 = document.querySelector('.header-get-function');
        if (typeof headerH1 !== 'undefined' && headerH1 !== null) {
            if(headerH1.classList.contains('home-header')){
                homeHeader.classList.add('home-header-container');
            }
        }
    }
});

$(function () {
    if(window.isAdBlockActive){
        let modalAdBlockerWarning = $('.modal.adblocker-warning');
        if(modalAdBlockerWarning.length > 0){
            $(modalAdBlockerWarning).modal('show');
        }
    }
});


//adBlockerIsActive
//adBlockerIsActive