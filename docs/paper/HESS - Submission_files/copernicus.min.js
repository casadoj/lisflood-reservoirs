$(function () {
    $('html').removeClass('no-js');
    $('.hide-js').css({'display': 'none'});


    $('.show-js').show();

    $('.mobile-citation').css('visibilty', 'hidden');
    $('.mobile-citation').addClass('mobile-device');

    $(document).on('click', '.special-issue-tabs a',function () {
        var allTabs = $('.special-issue-tabs a');
        allTabs.each(function () {
            $(this).removeClass('active');
        });

        $(this).addClass('active');
    });
    $(document).on('click touchend', '.show-hide',function (e) {
        var hideElementSelector = $(this).attr('data-hide');
        var showElementSelector = $(this).attr('data-show');
        var bubble = false;
        var duration = 0;

        if($(this).attr('data-duration') != undefined){
            duration = parseInt($(this).attr('data-duration'));
        }

        if ($(this).attr('data-bubble') != undefined) {
            bubble = $(this).attr('data-bubble') == 'true';
        }

        if(bubble === false){
            e.stopPropagation();
            e.preventDefault();
            e.stopImmediatePropagation();
        }
        var toggleElementSelector = $(this).attr('data-toggle');

        if (toggleElementSelector) {
            var toggleElement = $(toggleElementSelector);
            toggleElement.slideToggle(duration);

            return bubble;
        }


        var hideElement = $(hideElementSelector);


        var showElement = $(showElementSelector);

        showElement.fadeIn(duration);
        hideElement.hide({duration: duration});
        return bubble;
    });

    $(document).on('click touchend', '[data-toggle-required]', function (e) {
        let element = e.currentTarget;
        let requiredOn = $(element).attr('data-toggle-required') === 'on';
        let requiredOff = $(element).attr('data-toggle-required') === 'off';
        let cssSelector = $(element).attr('data-toggle-required-selector');
        let customAlertMessage = $(element).attr('data-toggle-required-alert');

        if (typeof cssSelector !== 'undefined' && cssSelector.length > 0) {
            let elementToToggleRequired = $(cssSelector);

            if (elementToToggleRequired.length > 0) {
                if (requiredOn) {
                    $(elementToToggleRequired).prop('required', true);
                    $(elementToToggleRequired).attr('oninvalid', 'alert("' + customAlertMessage + '")');
                    $(elementToToggleRequired).get(0).setCustomValidity(customAlertMessage);
                } else if (requiredOff) {
                    $(elementToToggleRequired).prop('required', false);
                    $(elementToToggleRequired).removeAttr('oninvalid');
                    $(elementToToggleRequired).get(0).setCustomValidity('');
                }

            }
        }

    });

    $(document).on('change', '[data-required]', function (e) {
        let element = e.currentTarget;
        let valueOfElement = $(element).val();

        if (valueOfElement.length > 0) {
            $(element).prop('required', false);
            $(element).removeAttr('oninvalid');
            $(element).get(0).setCustomValidity('');
        }
    });

    function handleMoreLess(that){
        var toggleCaption = that.attr('data-toggleCaption');
        var currentCaption = that.html();


        var hideElementSelector = that.attr('data-hide');
        var showElementSelector = that.attr('data-show');

        var hideElement = $(hideElementSelector);
        var showElement = $(showElementSelector);


        showElement.hide({duration: 0});
        hideElement.fadeIn();
        that.html(toggleCaption);

        that.attr('data-hide', showElementSelector);
        that.attr('data-show', hideElementSelector);

        if(toggleCaption === undefined){
            toggleCaption = currentCaption;
        }

        that.attr('data-toggleCaption', currentCaption);

        return false;
    }
    $(document).on('click','.more-less-mobile', function (e) {

        var width= $(document).width();

        if(width > 767) {
            return null;
        }

        return handleMoreLess($(this));
    });
    $(document).on('click','.more-less', function () {

        return handleMoreLess($(this));
    });

    $('body').on('click', '.figure-link,.table-link,.article-avatar, .paperlist-avatar,.table-download,.figure-download', function (e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        return false;
    });

    var pswpElement = document.querySelectorAll('.pswp')[0];
    var gallery = null;


    var articleAvatarElement = $('.article-avatar img');
    if(articleAvatarElement.length !== 0){
        var avatar = {
            src:articleAvatarElement.attr('data-web'),
            w:articleAvatarElement.attr('data-width'),
            h:articleAvatarElement.attr('data-height'),
            title:articleAvatarElement.attr('data-caption')
        };
        var avatarThumbnail = articleAvatarElement[0];
        articleAvatarElement.on('click',function(){
            var options = {
                showHideOpacity:true,
                bgOpacity:0.8,
                spacing:0.15,
                getThumbBoundsFn: function(index) {
                    // get window scroll Y
                    var pageYScroll = window.pageYOffset || document.documentElement.scrollTop;
                    // optionally get horizontal scroll

                    // get position of element relative to viewport
                    var rect = avatarThumbnail.getBoundingClientRect();

                    // w = width
                    return {x:rect.left, y:rect.top + pageYScroll, w:rect.width};


                    // Good guide on how to get element coordinates:
                    // http://javascript.info/tutorial/coordinates
                }

            };

            gallery = new PhotoSwipe( pswpElement, PhotoSwipeUI_Default,[avatar],options);

            gallery.init();
        });
    }
    var paperListAvatar = [];
    var paperListAvatarThumb = [];
    $('.paperlist-avatar img').each(function(){

        var webversion = $(this).attr('data-web');

        var width = $(this).attr('data-width');
        var height = $(this).attr('data-height');
        var caption =$(this).attr('data-caption');

        var figure = {
            src:webversion,
            w:width,
            h:height,
            title:caption
        };
        paperListAvatarThumb.push($(this)[0]);
        paperListAvatar.push(figure);
    });
    $('body').on('click', '.paperlist-avatar img', function (e) {
        if(paperListAvatarThumb.length === 0 && paperListAvatar.length === 0){
            $('.paperlist-avatar img').each(function(){

                var webversion = $(this).attr('data-web');

                var width = $(this).attr('data-width');
                var height = $(this).attr('data-height');
                var caption =$(this).attr('data-caption');

                var figure = {
                    src:webversion,
                    w:width,
                    h:height,
                    title:caption
                };
                paperListAvatarThumb.push($(this)[0]);
                paperListAvatar.push(figure);
            });
        }
        var target = $(this);

        var index = $('.paperlist-avatar img').index(target);

        var options = {
            showHideOpacity:true,
            bgOpacity:0.8,
            index:index,
            spacing:0.15,
            getThumbBoundsFn: function(index) {

                var thumbnail = paperListAvatarThumb[index];

                // get window scroll Y
                var pageYScroll = window.pageYOffset || document.documentElement.scrollTop;
                // optionally get horizontal scroll

                // get position of element relative to viewport
                var rect = thumbnail.getBoundingClientRect();

                // w = width
                return {x:rect.left, y:rect.top + pageYScroll, w:rect.width};


                // Good guide on how to get element coordinates:
                // http://javascript.info/tutorial/coordinates
            }

        };

        gallery = new PhotoSwipe( pswpElement, PhotoSwipeUI_Default,[paperListAvatar[index]],options);

        gallery.init();



    });

    $( window ).load(function() {

        if($('.auto-fixed-top').length > 0 && $('.auto-fixed-top').hasClass('filter') && $(window).width() < 1023){
            $('#mobile-nav .co-cogs').show();
            // $('.auto-fixed-top').hide();
            $('#mobile-nav-scrolled .co-cogs').show();

            $('#mobile-nav .co-cogs, #mobile-nav-scrolled .co-cogs').on('click', function (e) {
                if($('.auto-fixed-top').css('display') === 'none' || $('.auto-fixed-top').css('display') === ''){
                    $('.auto-fixed-top').show();
                } else {
                    $('.auto-fixed-top').hide();
                }

                if($('.auto-fixed-top').css('position') == 'fixed'){
                    $('.auto-fixed-top').css('top', '60px');
                } else {
                    $('.auto-fixed-top').css('top', '0');
                }
            });
        }

        $(window).on('scroll', function () {
            if($(window).width() < 1023){
                if($(window).scrollTop() >= 160 && ($('#w-body').outerHeight() - $('#w-head').outerHeight() > $(window).outerHeight() + $('#w-head').outerHeight())){
                    $('.auto-fixed-top:visible').hide();
                }
            }
        });

        if($('.auto-fixed-top').length === 0 || $('.auto-fixed-top').css('display') === 'none'){
            $(window).on('scroll', function () {
                if($(window).width() < 1023){
                    if($(window).scrollTop() >= 160 && ($('#w-body').outerHeight() - $('#w-head').outerHeight() > $(window).outerHeight() + $('#w-head').outerHeight())){
                        if($('.auto-fixed-top').length > 0 && $('.auto-fixed-top').hasClass('filter')) {
                            $('#mobile-nav-scrolled .co-cogs').show();
                        }
                        $("#mobile-search").addClass("mobile-search-fixed");
                        $("#j-mobile-banner").addClass("j-banner-fixed");
                        $("#j-topic").hide();
                        $("#j-secondary-nav").hide();
                        $('#mobile-nav-scrolled').css('display', 'block');
                        $('#j-mobile-banner #j-topic').hide();
                    } else {
                        $("#mobile-search").removeClass("mobile-search-fixed");
                        $("#j-mobile-banner").removeClass("j-banner-fixed");
                        $("#j-topic").show();
                        $("#j-secondary-nav").show();
                        $('#mobile-nav-scrolled').css('display', 'none');
                        $('#j-mobile-banner #j-topic').show();
                    }

                    if($('.auto-fixed-top').css('position') === 'fixed'){
                        $('.auto-fixed-top').css('top', '60px');
                    } else {
                        $('.auto-fixed-top').css('top', '0');
                    }
                } else {
                    $("#j-topic").show();
                    $("#j-secondary-nav").show();
                }
            })
        }
    });

    $('.co-mobile-menu').on('click', function (e) {
        setTimeout(function () {
            $('.active_menuitem:visible').closest('.menu_level2').css('visibility','visible');
            $('.active_menuitem:visible').closest('.menu_level2').css('height','auto');
            $('.active_menuitem:visible').closest('.menu_level2').closest('.menuitem_level1').css('height','auto');
        }, 100);
    });


    if($('#recent-template').length > 0){
        var journalUrl = $('#recent-template').data('journal-url');
        var journalShortCut = $('#recent-template').data('journal-shot-cut');
        $.getJSON(journalUrl + 'inc/' + journalShortCut + '/recent_papers.json',
            function (model) {
                return onRecentPapersJSON(model);
            }).error(function () {
            sendCriticalToCMS();
        });
    }

    if($('#highlight-template').length > 0){
        var jsonUrl = $('#highlight-template').data('json-url');
        $.getJSON(jsonUrl,
            function (model) {
                return onHighlightPapersJSON(model);
            }).error(function () {
            sendCriticalToCMS();
        });
    }

    function sendCriticalToCMS() {

    }

    function onRecentPapersJSON(models) {
        var view = $('#recent-template').html();
        var article = '';
        var numberOfPapers = $('#recent-paper-content').data('count');
        var projectShortCut = $('#recent-paper-content').data('short-cut');
        if(numberOfPapers > 0){
            models = models.slice(0, numberOfPapers);
        }
        $.each(models, function (id, model) {
            model.projectShortCut = projectShortCut;
            model.publishedDate = convertDate(model.publishedDate.date);
            article += Mustache.to_html(view, model);
        });
        if(article && article.length > 50){
            $("#recent-paper-content").html(article);
        }
    }

    function onHighlightPapersJSON(models) {
        var view = $('#highlight-template').html();
        var article = '';
        var numberOfPapers = $('#highlight-paper-content').data('count');
        var projectShortCut = $('#highlight-paper-content').data('short-cut');
        if(numberOfPapers > 0){
            models = models.slice(0, numberOfPapers);
        }
        $.each(models, function (id, model) {
            model.projectShortCut = projectShortCut;
            model.publishedDate = convertDate(model.publishedDate.date);
            article += Mustache.to_html(view, model);
        });
        if(article && article.length > 50){
            $("#highlight-paper-content").html(article);
        }
    }

    function convertDate(date){
        var options = { day: '2-digit', month: 'short', year: 'numeric'};
        date = new Date(date).toLocaleDateString('en-GB', options);
        return date;
    }

    /*---------------------mobile adopting stuff-------------------------------*/

    //displaying pdf-icon for articles

    if ($('.authors-short').length !== 0) {
        $('.co-mobile-pdf').css('display', 'initial');
    } else {
        $('.co-mobile-pdf').css('display', 'none');
    }

    // hiding/displaying left/right arrows in tabs in articles depends on width of tabs
    function checkResolution() {
        $('.mobile-citation').css('visibilty', 'hidden');
        $('.mobile-citation').addClass('mobile-device');
        var divWidth = $('.mobile-citation').innerWidth();
        var tabWidth = $('.mobile-citation .tab-navigation').prop('scrollWidth');

        if (divWidth >= tabWidth) {
            $('.mobile-citation').removeClass('mobile-device');
            $('.tab.co-angel-left').addClass('hidden-controls-tabs');
            $('.tab.co-angel-right').addClass('hidden-controls-tabs');
        } else {
            $('.mobile-citation').addClass('mobile-device');
            $('.tab.co-angel-left').removeClass('hidden-controls-tabs');
            $('.tab.co-angel-right').removeClass('hidden-controls-tabs');
        }

        $('.mobile-citation').css('visibilty', 'visible');
    }

    checkResolution();

    var width = screen.width,
        height = screen.height;
    var topArticlePanel = $('.auto-fixed-top');
    setInterval(function () {
        if (screen.width !== width || screen.height !== height) {
            width = screen.width;
            height = screen.height;
            $(window).trigger('resolutionchange');
        }
    }, 50);

    $(window).bind('resolutionchange', function (e) {
        setTimeout(() => checkResolution(), 100);
    });

    $(window).on('resize', function (e) {
        setTimeout(() => checkResolution(), 100);
    });

    var scrollToElement = null;
    if ($('li.active').length > 0){
        scrollToElement = $('li.active') || null;
    }

    if($('a.active_menuitem').length > 0) {
        scrollToElement = $('a.active_menuitem') || null;
    }

    if(scrollToElement !== null){
        var activeOffset = scrollToElement.offset().left;
        if ($('.mobile-citation .active').length > 0 && $('.co-angel-left').length > 0) {
            $(".mobile-citation").animate({
                scrollLeft: $('.mobile-citation .active').offset().left - $('.co-angel-left').width()
            }, 300);
        }
    }

    //animated scrolling of tabs in articles
    $(".tab.co-angel-left").click(function (e) {
        console.log(e.offsetX, e.target.offsetLeft);
        if (e.offsetX > e.target.offsetLeft) {
            $(".mobile-citation").animate({
                scrollLeft: '-=150'
            }, 800);
        }
    });
    $(".tab.co-angel-right").click(function (e) {
        console.log(e.offsetX, e.target.offsetLeft);
        if (e.offsetX < e.target.offsetLeft) {
            $(".mobile-citation").animate({
                scrollLeft: '+=150'
            }, 800);
        }

    });

    //displaying share icon
    if ($('#share-one-line').length > 0) {
        if ($('#share-one-line').css('display') === 'none') {
            $('.co-mobile-share:not(.share)').css('display', 'inline-block');
        } else {
            $('.co-mobile-share:not(.share)').css('display', 'none');

        }
    } else {
        $('.co-mobile-share:not(.share)').css('display', 'none');
    }

    if (!navigator.share) {
        $('.mobile-native-share:not(.share)').css('display', 'none');
    } else {
        $('#share-one-line').addClass('native-share-visible');
    }

    $('.mobile-native-share').click(function (e) {
        e.preventDefault();
        e.stopPropagation();
        if (navigator.share) {
            navigator.share({
                title: $(this).attr('data-title'),
                text: $(this).attr('data-text'),
                url: $(this).attr('data-url'),
            }).then(function () {
                console.log('Successful share');
                return false;
            })
                .catch(function (error) {
                    console.log('Error sharing', error);
                    return false;
                });
        }
        return false;
    });

    $('.co-mobile-share').click(function (e) {

        if ($(this).parent('a').attr('data-url') === undefined || $(this).parent('a').attr('data-url') === '') {
            var newShareBlock = $('#share-one-line').clone(true);
            newShareBlock.removeClass('hide-on-mobile');
            newShareBlock.removeClass('hide-on-tablet');
            newShareBlock.addClass('co-mobile-share-block');
            newShareBlock.removeAttr('style');
            newShareBlock.css('top', '50%');
            $('body').addClass('noscroll');
            $('body').append('<div class="co-mobile-overlay"></div>').append(newShareBlock);
        }
    });

    //hiding share icon
    $(document).on('click', '.co-mobile-overlay', function (e) {
        $('.co-mobile-overlay').remove();
        $('.co-mobile-share-block').remove();
        $('body').removeClass('noscroll');
    });

    //downloading pdf for articles
    var linkToPdf = $('.pdf-icon').attr('href');
    if($('.co-mobile-pdf').length > 0){
        $('.co-mobile-pdf').parent('a').attr('href', linkToPdf);
    }

    //Error messages anchor
    $(document).on('click', '.error-message[data-target]', function (event) {
        var element = $(this);
        var target = $(element).data('target');
        var targetClickElementSelector = $(element).data('target-click-element');
        var targetClickElement = $(target).find(targetClickElementSelector);

        var elementScrollTo = $(target);

        if(elementScrollTo.length > 0){
            var offset = $(elementScrollTo).offset();
            $('body,html').animate({scrollTop: offset.top-150}, 300);
        }

        if(targetClickElement.length > 0){
            $(targetClickElement).trigger('click');
        }
    });

    //Share
    $(document).on('click', '.desktop-share', function (event) {
        event.preventDefault();
        event.stopPropagation();
        let element = $(this);
        let linkValue = $(element).data('href');

        updateClipboard(linkValue, element);

        return false;
    });

    /**
     * Method for copy something to clipboard
     * @param newClip
     * @param element
     */
    window.updateClipboard = function (newClip, element) {
        navigator.clipboard.writeText(newClip).then(function () {

        }, function (err) {
            console.warn('Error during copying to clipboard: ', err.toString());
        });
    };

    /**
     *
     * @param event
     * @param element
     * @returns {boolean}
     */
    window.nativeShare = function (event, element) {
        event.preventDefault();
        event.stopPropagation();
        if (navigator.share) {
            navigator.share({
                title: $(element).attr('data-title'),
                text: $(element).attr('data-text'),
                url: $(element).attr('data-url'),
            }).then(function () {
                console.log('Successful share');
                return false;
            })
                .catch(function (error) {
                    console.error('Error sharing', error);
                    return false;
                });
        } else {
            let linkValue = $(element).data('href');
            updateClipboard(linkValue, element);
        }
        return false;
    };
});
