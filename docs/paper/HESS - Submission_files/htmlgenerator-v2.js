window.onlyClassification = false;
window.msTypePreselectedOnload = false;
$(function () {
    $('html').removeClass('no-js');
    $('.hide-js').css({'display': 'none'});

    try {
        $('[data-toggle="popover"]').popover();
    } catch (error) {
        console.log(error);
    }

    $('.show-js').show();
    $('#contentFilter,#presentationTypeFilter').on('click', 'label', function () {
        let presentationTypeFilter = $('input[name="presentationTypeFilter"]:checked');
        let contentFilter = $('input[name="contentFilter"]:checked');
        let cssClass = '';

        if (contentFilter.length > 0) {
            cssClass = cssClass + '.' + contentFilter.val().trim();
        }
        if (presentationTypeFilter.length > 0) {
            cssClass = cssClass + '.' + presentationTypeFilter.val().trim();
        }


        $('.abstractListObject').removeClass('filtered').hide();
        $(cssClass).addClass('filtered').show();
        $('body').trigger('updateAbstractContainer');
    });
    $('body').on('updateAbstractContainer', function () {
        $('.abstractContainer').each(function () {
            let selector = '.' + $(this).attr('id');
            let element = $(this).find('.abstractListObject');


            if (element.length === 0) {
                return true;
            }
            let hasAbstractListObjects = $(this).find('.filtered').length > 0;
            if (!hasAbstractListObjects) {

                $(this).hide();


                $(selector).hide();
            } else {
                if ($(this).is(':visible')) {
                    $(this).show();
                }

                $(selector).show();
            }

            $(this).find('h3').each(function () {
                let selector = $(this).attr('id');
                let selectorString = '.session-' + selector + '.filtered';

                let abstractsPerSession = $(selectorString).length > 0;
                if (!abstractsPerSession) {
                    $(this).hide();
                } else {

                    $(this).show();
                }

            });
        });
    });
    var updateList = function (min, max) {
        var selectedTypeValues = [];


        if (min == undefined && max == undefined && timeRange !== undefined) {
            var timeRangeValues = timeRange.slider('option', 'values')
            min = timeRangeValues[0];
            max = timeRangeValues[1];

            if (min == undefined && max == undefined) {
                min = 0;
                max = 0;
            }
            min = labelValues[min];
            max = labelValues[max];

        } else {
            min = labelValues[min];
            max = labelValues[max];
        }

        $('.classification input:not(:checked)').each(function () {
            var selector = '.discussion-results > .paperlist-object.' + $(this).val() + ':visible';
            selectedTypeValues.push(selector);

            //$(selector).hide();
        });
        $(selectedTypeValues.join()).hide();

        selectedTypeValues = [];
        var selectedTypes = [];
        $('.classification input:checked').each(function () {
            var selector = '.discussion-results > .paperlist-object.' + $(this).val();
            selectedTypes.push($(this).val());
            if (min !== 0 && max !== 0) {
                selector += ':visible';
            }
            // $(selector).show();
            selectedTypeValues.push(selector);
        });


        $(selectedTypeValues.join()).show();

        if (!window.onlyClassification) {
            $('.paperlist-object').each(function () {
                var diff = parseInt($(this).attr('data-diff'));

                if (diff >= min && diff <= max) {
                    let showItem = false;
                    for (let i = 0; i < selectedTypes.length; i++) {
                        let selectedType = selectedTypes[i];
                        if ($(this).hasClass(selectedType)) {
                            showItem = true;
                        }
                    }
                    if (showItem) {
                        $(this).addClass('in-range').show();
                    }
                } else {
                    $(this).removeClass('in-range').hide();
                }

            });
        } else {
            $('.paperlist-object').each(function () {
                if (!$(this).hasClass('in-range')) {
                    $(this).hide();
                }
            });
        }

        $(document).trigger('check-messages');
    };
    $('.video-abstract-thumb a').on('click', function (e) {
        var iframe = $(this).closest('.video-abstract-frame-content').find('iframe');
        $(this).parent('.video-abstract-thumb').hide();
        var url = $(this).attr('href');

        iframe.attr('src', url);
        iframe.show();

        iframe.trigger('click', e);
        return false;
    });
    $('ul.tabnav li,ul.tab-navigation li').on('click', function () {

        var a = $(this).find('a');

        if (a.attr('href') == '#') {

            $('.tab-content').hide({duration: 0});
            $('ul.tabnav li.active,ul.tab-navigation li').removeClass('active');
            $(this).addClass('active');
            var tabname = a.text().toLowerCase();
            $('.tab-content.' + tabname).show();

            if (tabname == 'metrics' && categories.length > 0) {
                var model = CoPublisher.JournalMetrics.Model;
                model.init(categories, series);

                var view = CoPublisher.JournalMetrics.View;
                view.init("#highcharts-container", model, config);
                view.render();


                var model = CoPublisher.JournalMetrics.Model;
                model.init(categories, seriesCumulative);

                var view = CoPublisher.JournalMetrics.View;
                view.init("#highcharts-container-cumulative", model, config);
                view.render();

                if (typeof resizeDraw == 'function') {
                    resizeDraw();

                }
                if (typeof renderMapOld == 'function') {
                    renderMapOld('#map-complete', completeData, completetTotal);
                }

            }
            return false;
        }

    });
    const anchor = window.location.hash.substring(1);
    const regex = /(CC|EC|CEC|AC|RC)/;
    let match;
    let isComment = (match = regex.exec(anchor)) !== null;

    if (anchor.includes('discussion') ||
        isComment
    ) {

        var discussionTab = $('ul.tabnav li a[href*="#"]:contains("Discussion")').parent('li').trigger('click');
        $('ul.tab-navigation li a[href*="#"]:contains("Discussion")').closest('li').trigger('click');
        if (isComment) {
            let commentBody = $('.' + anchor);

            commentBody.show();
        }
    }

    var loader = $('.ajax-loader');
    var lightbox = $('.lightbox');
    $('form.ajax-form').submit(function (e) {
        e.preventDefault();
        e.stopPropagation();
        var form = $(this);

        var actionString = form.attr('action');
        //var dataArray = form.serialize();

        var methodString = form.attr('method');
        var targetString = form.attr('target');
        let useCredentials = form.attr('data-credentials') === undefined;
        let options = {
            method: methodString,
            url: actionString,
            data: new FormData(this),
            processData: false,
            contentType: false,
            crossDomain: true,

            headers: {'X-Requested-With': 'XMLHttpRequest'},
            beforeSend: function (xhr, settings) {
                form.trigger('ajax-form-before', [xhr, targetString, form]);
            }
        }

        if (useCredentials) {

            options.xhrFields = {
                withCredentials: true
            }
        }

        loader.show();
        lightbox.show();
        $.ajax(options).done(function (result) {
            loader.hide();
            lightbox.hide();
            form.trigger('ajax-form-done', [result, targetString, form]);
        }).fail(function (result) {
            loader.hide();
            lightbox.hide();
            form.trigger('ajax-form-fail', [result, targetString, form]);
        });


        return null;
    });

    $('.ajax-content').each(function () {
        let dataSrc = $(this).attr('data-src');

        let contentBox = $(this);
        loader.show();
        lightbox.show();
        $.ajax({
            method: 'GET',
            url: dataSrc,
            crossDomain: true,
            xhrFields: {
                withCredentials: true
            },
            headers: {'X-Requested-With': 'XMLHttpRequest'},
            beforeSend: function (xhr, settings) {
                contentBox.trigger('ajax-content-before', [xhr, dataSrc, contentBox]);
            }
        }).done(function (result) {
            loader.hide();
            lightbox.hide();
            try {
                contentBox.html(result);
            } catch (error) {
                console.log(error);
            }
            contentBox.trigger('ajax-content-done', [result, dataSrc, contentBox]);
        }).fail(function (result) {
            loader.hide();
            loaderlightbox.hide();
            contentBox.trigger('ajax-content-fail', [result, dataSrc, contentBox]);
        });

    });

    $('body').on('ajax-content-done', '#discussionHTML', function (event, result, targetUrl, contentBox) {

        if (isComment) {
            let commentBody = $('.' + anchor);
            commentBody.show();
        }
    });

    $('body').on('ajax-form-done', '#ajaxSearch', function (e, result, target, form) {
        $('#' + target).html(result);
    });
    $('body').on('ajax-form-done', '#searchForm', function (e, result, target, form) {

        var view = $('#searchResultTemplate').html();
        var paperListObjectView = $('#paperListObjectTemplate').html();
        var specialIssueView = $('#specialIssueTemplate').html();
        var volumeView = $('#volumeTemplate').html();
        var issueView = $('#issueTemplate').html();
        var html = Mustache.render(view, result, {
            'htmlgenerator/partials/paperListObject': paperListObjectView,
            'htmlgenerator/partials/specialIssue': specialIssueView,
            'htmlgenerator/partials/volume': volumeView,
            'htmlgenerator/partials/issue': issueView
        });


        $(target).html(html);

    });
    var closed = true;
    $('.pdf-icon.show-hide').on('click', function () {

        closed = !closed;
        if (!closed) {
            $(this).css('opacity', 0.4);
        } else {
            $(this).css('opacity', 1);
        }
    });

    $('[data-marker-name]').on('click', function () {
        var markerName = $(this).attr('data-marker-name');

        if (markerName != undefined) {
            $('[data-marker-name="' + markerName + '"]').removeClass('active');
            $(this).addClass('active');
        }
    });


    $(document).on('click', 'h2.open,h2.closed,.h1.open,.h1.closed', function () {
        var isOpened = $(this).hasClass('open');
        var contentBox = $(this).next('div');
        if (contentBox.length === 0) {
            contentBox = $(this).closest('div').next('div');
        }


        if (isOpened) {
            $(this).removeClass('open').addClass('closed');
            contentBox.removeClass('open').addClass('closed');

        } else {
            $(this).removeClass('closed').addClass('open');
            contentBox.removeClass('closed').addClass('open');
        }
    });

    $('.viewedSwitch input[type="radio"]').on('change', function () {
        var value = $(this).val();
        var tmpCategories = categoriesSwitch[value];
        var tmpSeries = seriesSwitch[value];
        var tmpCumulativeSeries = cumulativeSeriesSwitch[value];
        var currentSelector = '.metrics-viewed.' + value;
        $('.metrics-viewed').css({'display': 'none'});
        $(currentSelector).show(0);
        var model = CoPublisher.JournalMetrics.Model;
        model.init(tmpCategories, tmpSeries);

        var view = CoPublisher.JournalMetrics.View;
        view.init(currentSelector + " #highcharts-container", model, config);
        view.render();


        var model = CoPublisher.JournalMetrics.Model;
        model.init(tmpCategories, tmpCumulativeSeries);

        var view = CoPublisher.JournalMetrics.View;
        view.init(currentSelector + " #highcharts-container-cumulative", model, config);
        view.render();

    });
    $('.citedSwitch input[type="radio"]').on('change', function () {
        var value = $(this).val();
        var currentSelector = '.metrics-cited.' + value;
        $('.metrics-cited').css({'display': 'none'});
        $(currentSelector).css({'display': 'inline-block'});
    });

    var pswpElement = document.querySelectorAll('.pswp')[0];
    var gallery = null;
    var figures = [];
    var figureThumbs = [];

    $('.fig a.figure-link img').each(function () {

        var webversion = $(this).attr('data-webversion');


        if (typeof webversion === "undefined" || webversion === '') {
            return false;
        }


        var width = $(this).attr('data-width');
        var height = $(this).attr('data-height');
        var figure = {
            src: webversion,
            w: width,
            h: height,
            title: ''
        };

        figureThumbs.push($(this)[0]);
        figures.push(figure);

    });
    var tables = [];
    var tableThumbs = [];
    $('.table-wrap img[data-webversion]').each(function () {

        var webversion = $(this).attr('data-webversion');


        if (typeof webversion === "undefined" || webversion === '') {
            return false;
        }

        var width = $(this).attr('data-width');
        var height = $(this).attr('data-height');

        var table = {
            src: webversion,
            w: width,
            h: height,
            title: ''
        };
        tableThumbs.push($(this)[0]);
        tables.push(table);

    });

    $('.table-wrap').on('click', function (e) {
        var target = $(this);
        var caption = target.find('.caption').html();

        var newTarget = target.find('img[data-webversion]');
        if (newTarget.length > 0) {
            target = newTarget;
        }
        var webversion = target.attr('data-webversion');
        var printVersion = target.attr('data-printversion');
        var csvVersion = target.attr('data-csvversion');
        if (typeof webversion === "undefined" || webversion === '') {
            return false;
        }

        var tableFooter = target.closest('.table-wrap').find('.table-wrap-foot');

        if (tableFooter.length > 0) {
            caption += tableFooter.html();
        }

        var title = '<span class="captionLong">' + caption + '</span>';
        var documentWidth = $('body').width();
        var needShortVersion = $(caption).text().length > 50 && documentWidth <= 767;
        if (needShortVersion) {
            var title = '<div class="captionLong" style="display: none">' + caption + '</div><div class="captionShort">' + caption + '</div>';

            title += '<a  href="#" data-bubble="false" data-hide=".moreButton,.captionShort" data-show=".captionLong,.hideButton" class="show-hide journal-contentLinkColor triangle moreButton">Read more</a>';
            title += '<a  href="#" data-bubble="false" data-show=".moreButton,.captionShort" data-hide=".captionLong,.hideButton" style="display: none" class="show-hide journal-contentLinkColor triangle hideButton">Hide</a>';
        }

        if (printVersion !== '' && printVersion !== undefined) {
            title += '<a href="' + printVersion + '" target="_blank" class="triangle journal-contentLinkColor table-download">Download Print Version</a>'
        }
        if (csvVersion !== '' && csvVersion !== undefined) {
            title += '<a href="' + csvVersion + '" target="_blank" class="triangle journal-contentLinkColor table-download">Download XLSX</a>'
        }

        var index = $('.table-wrap img[data-webversion]').index(target)
        tables[index].title = title;
        var options = {
            showHideOpacity: true,
            bgOpacity: 0.8,
            index: index,

            getThumbBoundsFn: function (index) {

                var thumbnail = tableThumbs[index];

                // get window scroll Y
                var pageYScroll = window.pageYOffset || document.documentElement.scrollTop;
                // optionally get horizontal scroll

                // get position of element relative to viewport
                var rect = thumbnail.getBoundingClientRect();

                // w = width
                return {x: rect.left, y: rect.top + pageYScroll, w: rect.width};


                // Good guide on how to get element coordinates:
                // http://javascript.info/tutorial/coordinates
            }

        };

        gallery = new PhotoSwipe(pswpElement, PhotoSwipeUI_Default, [tables[index]], options);

        gallery.init();


    });
    $('.fig').on('click', function (e) {
        var target = $(this);
        var caption = target.find('.caption').html();

        var newTarget = target.find('img');
        if (newTarget.length > 0) {
            target = newTarget;
        }
        var webversion = target.attr('data-webversion');
        var printVersion = target.attr('data-printversion');

        if (typeof webversion === "undefined" || webversion === '') {

            return false;
        }


        var index = $('.fig a.figure-link img').index(target);

        var title = '<span class="captionLong">' + caption + '</span>';
        var documentWidth = $('body').width();
        var needShortVersion = $(caption).text().length > 50 && documentWidth <= 767;
        caption = '<span>' + caption + '</span>';
        if (needShortVersion) {

            var title = '<div class="captionLong" style="display: none">' + caption + '</div><div class="captionShort">' + caption + '</div>';

            title += '<a  href="#" data-bubble="false" data-hide=".moreButton,.captionShort" data-show=".captionLong,.hideButton" class="show-hide journal-contentLinkColor triangle moreButton">Read more</a>';
            title += '<a  href="#"  data-bubble="false" data-show=".moreButton,.captionShort" data-hide=".captionLong,.hideButton" style="display: none" class="show-hide journal-contentLinkColor triangle hideButton">Hide</a>';
        }

        if (printVersion !== '' && printVersion !== undefined) {
            title += '<a href="' + printVersion + '" target="_blank" class="triangle journal-contentLinkColor figure-download">Download</a>'
        }

        figures[index].title = title;

        var options = {
            showHideOpacity: true,
            bgOpacity: 0.8,
            index: index,

            getThumbBoundsFn: function (index) {

                var thumbnail = figureThumbs[index];

                // get window scroll Y
                var pageYScroll = window.pageYOffset || document.documentElement.scrollTop;
                // optionally get horizontal scroll

                // get position of element relative to viewport
                var rect = thumbnail.getBoundingClientRect();

                // w = width
                return {x: rect.left, y: rect.top + pageYScroll, w: rect.width};


                // Good guide on how to get element coordinates:
                // http://javascript.info/tutorial/coordinates
            }

        };

        gallery = new PhotoSwipe(pswpElement, PhotoSwipeUI_Default, [figures[index]], options);

        gallery.init();


    });

    function getOffset(el) {
        el = el.getBoundingClientRect();

        var y = window.scrollY || window.pageYOffset;
        var x = window.scrollX || window.pageXOffset;
        return {
            left: el.left + x,
            top: el.top + y
        }
    }

    $('.scrollto').on('click', function (e) {
        var fixedElementSelector = $(this).attr('data-fixed-element');
        var fixedElement = $(fixedElementSelector);
        var elementToShow = $(this).attr('data-element-toggle');

        e.stopPropagation();
        var href = $(this).attr('href');
        var hash = href.substr(href.indexOf('#') + 1);

        var element = document.getElementById(hash);
        if (element === undefined) {
            return true;
        }
        var target = $('#' + hash.replace('.', '\\.'));

        if (!target.is(':visible')) {
            target.closest('.sec').find('.more-less-mobile').trigger('click', e);

        }
        if (target.find('.h1:visible').length > 0) {
            target = target.find('.h1:visible');
        }

        var topPosition = ~~(target.offset().top);

        if (fixedElement !== undefined) {
            var height = fixedElement.outerHeight();
            if (fixedElement.length > 1) {
                for (var i = 0, il = fixedElement.length; i < il; i++) {
                    var current = fixedElement.eq(i);
                    height = current.outerHeight();
                    if (height > 0) {
                        break;
                    }
                }
                fixedElement = current;
            }

            topPosition -= height;

            var oldElement = fixedElement.clone();
        }

        topPosition -= 5;

        if (typeof elementToShow !== 'undefined' && elementToShow.length > 0 && !$(elementToShow).is(':visible')) {
            $(elementToShow).slideDown(100);
        }

        $('body,html').animate({scrollTop: topPosition}, 500, function () {

            if (fixedElement !== undefined && oldElement.css('position') !== 'fixed') {

                topPosition = ~~(target.offset().top);

                topPosition -= fixedElement.outerHeight();
                topPosition -= 5;

                $('body,html').animate({scrollTop: topPosition}, 500);
            }
        });

    });
    var fixedElement = $('.auto-fixed-top,.auto-fixed-top-forced');

    fixedElement.on('toggleFixed', function (event, e) {
        var fixedElement = $(this);
        var offsetTop = e.currentTarget.pageYOffset;

        if (fixedElement.attr('data-topline') === undefined) {
            let top = ~~(fixedElement.offset().top);
            if (fixedElement.attr('data-fixet-top-target') !== undefined) {
                if ($(fixedElement.attr('data-fixet-top-target')).length > 0) {
                    top = ~~($(fixedElement.attr('data-fixet-top-target')).offset().top);
                }
            }

            if (fixedElement.closest('.isprs-2020-wrapper').length !== 0) {
                top = 600;
            }
            fixedElement.attr('data-topline', top);
        }
        var topLine = fixedElement.attr('data-topline');
        var skipNextElementStyling = fixedElement.attr('data-skip-next') || 0;
        var nextElement = false;
        if (parseInt(skipNextElementStyling) !== 1) {
            nextElement = fixedElement.next('div');
        }
        var hiddenElements = $('.show-on-fixed');

        var shownElements = $('.hide-on-fixed');
        var noShadow = fixedElement.hasClass('no-shadow');
        if (offsetTop !== 0 && offsetTop >= topLine) {

            if (fixedElement.css('position') !== 'fixed') {

                hiddenElements.show();
                shownElements.hide();
                if (!noShadow && !$(fixedElement).hasClass('articleNavigation')) {
                    fixedElement.addClass('shadow');
                }

                fixedElement.css({
                    'position': 'fixed',
                    'opacity': 0.5
                }).addClass('fixed').animate({opacity: 1}, 500);
                var height = fixedElement.outerHeight();
                if(nextElement !== false){
                    nextElement.css({'margin-top': height + 'px'});
                }
            }
            var width = $(fixedElement.parents('div')[0]).width();

            fixedElement.css({'width': width + 'px'});
        } else {
            hiddenElements.hide();
            shownElements.show();
            if (!noShadow) {
                fixedElement.removeClass('shadow');
            }
            fixedElement.css({'position': 'relative', 'width': '100%'}).removeClass('fixed');
            if(nextElement !== false) {
                nextElement.css({'margin-top': '0px', 'padding-top': '0px'});
            }

        }
    });

    function updateFixedElement(e) {
        fixedElement.trigger('toggleFixed', [e]);
    }

    $(window).on('scroll', updateFixedElement).resize(function (e) {

        updateFixedElement(e);
    });

    var timeRange = $('#time-range');
    if (timeRange.length != 0) {

        $('.classification').on('change', 'input', function (event) {
            window.onlyClassification = true;
            updateList();
            if (window.msTypePreselected && !window.msTypePreselectedOnload) {
                let prevContent = $('.home-header').attr('data-prev-content');
                $('.home-header').html(prevContent);
            }
            $(document).trigger('check-messages');
        });

        $(document).on('check-messages', function () {
            if ($('.paperlist-object:visible').length == 0) {
                $('.empty-list-message').show();
            } else {
                $('.empty-list-message').hide();
            }
            if ($('.classification input').length != 0 && $('.classification input:checked').length == 0) {
                $('.category-message').show();
                $('.empty-list-message').hide();
            } else {
                $('.category-message').hide();
            }
        });


        timeRange.slider({
            range: true,
            min: 0,
            max: labels.length - 1,
            step: 1,
            values: [defaultStart || 0, defaultEnd],
            slide: function (event, ui) {
                if (ui.values[0] == ui.values[1]) {
                    return false;
                }
                $(document).trigger('check-messages');
            },
            change: function (event, ui) {
                window.onlyClassification = false;
                updateList(ui.values[0], ui.values[1]);
                $(document).trigger('check-messages');
                $(document).trigger('timeRange.changed', ui);


            },
            create: function (event, ui) {
                updateList();
                $(document).trigger('check-messages');
            }
        }).slider("pips", {
            labels: labels,
            rest: "label"
        });

        timeRange.find('.ui-slider-label').each(function (index) {
            var text = titles[index];
            $(this).attr('title', text);

        });
        if ((labels.length) == 2) {
            //timeRange.slider('destroy');
            timeRange.hide();
        }


        updateList();
    }

    $(document).on('renderMap', function (event, selector, data, total) {

        var highestNumber = 0;
        var domain = [];
        var topFiveCountries = {};
        var counter = 0;
        var totalSum = total;
        var topFiveCountriesSelector = selector.replace(".map", "").replace("#", ".").trim() + ".top-countries";
        var topFiveCountriesDiv = $(topFiveCountriesSelector);

        var countryDetailsDiv = $(selector + " .country-details");
        var legendDiv = $(selector + " .country-legend");
        var topFiveCountriesBody = topFiveCountriesDiv.find('tbody');


        for (var i in data) {
            var currentRow = data[i];
            var maxNumberInRow = d3.max(currentRow);
            if (maxNumberInRow > highestNumber) {
                highestNumber = maxNumberInRow;
            }

            if (counter < 5) {
                if (i !== "UNKNOWN" || i !== null || i !== undefined) {
                    topFiveCountries[i] = currentRow;
                }
            }
            counter++;
            domain.push(currentRow[0]);
        }
        legendDiv.find('.end').text(highestNumber);
        $(selector).next('.country-legend').find('.end').text(highestNumber);
        var colors = [
            0, 1, 2, 3, 4
        ];

        $(selector).find('svg').remove();
        var scale = d3.scale.quantile()

            .domain(domain)
            .range(colors);

        var log = d3.scale.log().domain([1, highestNumber]).range([0, colors.length - 1]);


        var width = $(selector).innerWidth();
        var height = width / 2;
        var offsetLeft = 0;
        var offsetTop = 0;


        var active = d3.select(null);
        var projection = d3.geo.equirectangular()
            .rotate([-6, 0, 0])
            .scale(width / 2 / Math.PI)
            .translate([(width / 2) + offsetLeft, (height / 2) + offsetTop]);

        var zoom = d3.behavior.zoom()
            .translate([offsetLeft, offsetTop])
            .scale(1)
            .scaleExtent([1, 8])
            .on("zoom", zoomed);

        var path = d3.geo.path()
            .projection(projection);

        var svg = d3.select(selector).append("svg")
            .attr("width", width)
            .attr("height", height)
            .on("click", stopped, true);

        svg.append("rect")
            .attr("class", "background")
            .attr("width", width)
            .attr("height", height)
            .on("click", reset);

        var g = svg.append("g");


        svg.call(zoom.event); //initial zoom

        var geoData = topojson.feature(worldJson, worldJson.objects.world).features;
        /* geoData = geoData.filter(function (feature) {
          //   return feature.id !== "ATA"; //exclude Antarctica
         });*/
        var mappedCountryNames = {};
        for (var i in geoData) {
            var current = geoData[i];
            mappedCountryNames[current.id] = current.properties.name;
            //  console.log(current.id);
        }

        counter = 1;
        topFiveCountriesBody.html("");
        for (var i in topFiveCountries) {
            var currentCountry = topFiveCountries[i];
            var name = mappedCountryNames[i];
            var percent = ~~((currentCountry[0] / totalSum) * 100);
            var html = '<tr>' +
                '<td class="country">' + name + '</td>' +
                '<td class="rank">' + counter + '</td>' +
                '<td class="views">' + currentCountry[0] + '</td>' +
                '<td class="percent">' + percent + '</td>' +
                '</tr>';
            counter++;

            topFiveCountriesBody.append(html);
        }


        g.selectAll("path")
            .data(geoData)
            .enter().append("path")
            .attr("d", path)
            .attr("class", function (d) {
                var className = 'feature ' + d.id;

                if (data[d.id] !== undefined) {
                    var currentRow = data[d.id];
                    //console.log(scale(currentRow[0]));
                    className += " color-range-" + ~~log(currentRow[0]);
                }
                return className;
            })
            .attr('popup', function (d) {

            })
            .attr('name', function (data) {
                return data.properties.name;
            })

            .on("click", clicked);


        function clicked(d) {
            if (active.node() === this) return reset();
            active.classed("active", false);
            active = d3.select(this).classed("active", true);
            // topFiveCountriesDiv.hide();
            countryDetailsDiv.show();
            var total = 0, html = 0, pdf = 0, xml = 0, name = active.attr('name'); //mappedCountryNames[d.id];
            if (data[d.id] !== undefined) {
                var currentRow = data[d.id];
                total = currentRow[0];
                html = currentRow[1];
                pdf = currentRow[2];
                xml = currentRow[3];
            }
            countryDetailsDiv.find('th').text(name);
            countryDetailsDiv.find('.total .value').text(total);
            countryDetailsDiv.find('.html .value').text(html);
            countryDetailsDiv.find('.pdf .value').text(pdf);
            countryDetailsDiv.find('.xml .value').text(xml);

            var bounds = path.bounds(d),
                dx = bounds[1][0] - bounds[0][0],
                dy = bounds[1][1] - bounds[0][1],
                x = (bounds[0][0] + bounds[1][0]) / 2,
                y = (bounds[0][1] + bounds[1][1]) / 2;

            if (d.id === "RUS") {

                dx = 400;
                dy = 50;
                x = 650;
                y = 80;

            }

            var scale = Math.max(1, Math.min(8, 0.9 / Math.max((dx) / width, (dy) / height))),
                translate = [(width / 2) + offsetLeft - scale * x, (height / 2) + offsetTop - scale * y];
            svg.transition()
                .duration(500)
                .call(zoom.translate(translate).scale(scale).event);
        }

        function reset() {
            active.classed("active", false);
            active = d3.select(null);

            svg.transition()
                .duration(500)
                .call(zoom.translate([offsetLeft, offsetTop]).scale(1).event);
            // topFiveCountriesDiv.show();
            countryDetailsDiv.hide();
        }

        function zoomed() {
            g.style("stroke-width", 1.5 / d3.event.scale + "px");
            g.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
        }


        function stopped() {
            if (d3.event.defaultPrevented) d3.event.stopPropagation();
        }

    });

    function renderMap(selector, data, total) {


        var highestNumber = 0;
        var domain = [];
        var topFiveCountries = {};
        var counter = 0;
        var totalSum = total;

        var topFiveCountriesDiv = $(selector.replace("#", ".") + ".top-countries");
        var countryDetailsDiv = $(selector + " .country-details");
        var legendDiv = $(selector + " .country-legend");
        var topFiveCountriesBody = topFiveCountriesDiv.find('tbody');


        for (var i in data) {
            var currentRow = data[i];
            var maxNumberInRow = d3.max(currentRow);
            if (maxNumberInRow > highestNumber) {
                highestNumber = maxNumberInRow;
            }

            if (counter < 5) {
                if (i !== "UNKNOWN" || i !== null || i !== undefined) {
                    topFiveCountries[i] = currentRow;
                }
            }
            counter++;
            domain.push(currentRow[0]);
        }
        legendDiv.find('.end').text(highestNumber);
        var colors = [
            0, 1, 2, 3, 4
        ];


        var scale = d3.scale.quantile()

            .domain(domain)
            .range(colors);

        var log = d3.scale.log().domain([1, highestNumber]).range([0, colors.length - 1]);


        var width = 842;
        var height = 400;
        var offsetLeft = 0;
        var offsetTop = 0;


        var active = d3.select(null);
        var projection = d3.geo.equirectangular()
            .rotate([-6, 0, 0])
            .scale(width / 2 / Math.PI)
            .translate([(width / 2) + offsetLeft, (height / 2) + offsetTop]);

        var zoom = d3.behavior.zoom()
            .translate([offsetLeft, offsetTop])
            .scale(1)
            .scaleExtent([1, 8])
            .on("zoom", zoomed);

        var path = d3.geo.path()
            .projection(projection);

        var svg = d3.select(selector).append("svg")
            .attr("width", width)
            .attr("height", height)
            .on("click", stopped, true);

        svg.append("rect")
            .attr("class", "background")
            .attr("width", width)
            .attr("height", height)
            .on("click", reset);

        var g = svg.append("g");


        svg.call(zoom.event); //initial zoom

        var geoData = topojson.feature(worldJson, worldJson.objects.world).features;
        /* geoData = geoData.filter(function (feature) {
          //   return feature.id !== "ATA"; //exclude Antarctica
         });*/
        var mappedCountryNames = {};
        for (var i in geoData) {
            var current = geoData[i];
            mappedCountryNames[current.id] = current.properties.name;
            //  console.log(current.id);
        }

        counter = 1;

        for (var i in topFiveCountries) {
            var currentCountry = topFiveCountries[i];
            var name = mappedCountryNames[i];
            var percent = ~~((currentCountry[0] / totalSum) * 100);
            var html = '<tr>' +
                '<td class="country">' + name + '</td>' +
                '<td class="rank">' + counter + '</td>' +
                '<td class="views">' + currentCountry[0] + '</td>' +
                '<td class="percent">' + percent + '</td>' +
                '</tr>';
            counter++;

            topFiveCountriesBody.append(html);
        }


        g.selectAll("path")
            .data(geoData)
            .enter().append("path")
            .attr("d", path)
            .attr("class", function (d) {
                var className = 'feature ' + d.id;

                if (data[d.id] !== undefined) {
                    var currentRow = data[d.id];
                    //console.log(scale(currentRow[0]));
                    className += " color-range-" + ~~log(currentRow[0]);
                }
                return className;
            })
            .attr('popup', function (d) {

            })
            .attr('name', function (data) {
                return data.properties.name;
            })

            .on("click", clicked);


        function clicked(d) {
            if (active.node() === this) return reset();
            active.classed("active", false);
            active = d3.select(this).classed("active", true);
            // topFiveCountriesDiv.hide();
            countryDetailsDiv.show();
            var total = 0, html = 0, pdf = 0, xml = 0, name = active.attr('name'); //mappedCountryNames[d.id];
            if (data[d.id] !== undefined) {
                var currentRow = data[d.id];
                total = currentRow[0];
                html = currentRow[1];
                pdf = currentRow[2];
                xml = currentRow[3];
            }
            countryDetailsDiv.find('th').text(name);
            countryDetailsDiv.find('.total .value').text(total);
            countryDetailsDiv.find('.html .value').text(html);
            countryDetailsDiv.find('.pdf .value').text(pdf);
            countryDetailsDiv.find('.xml .value').text(xml);

            var bounds = path.bounds(d),
                dx = bounds[1][0] - bounds[0][0],
                dy = bounds[1][1] - bounds[0][1],
                x = (bounds[0][0] + bounds[1][0]) / 2,
                y = (bounds[0][1] + bounds[1][1]) / 2;

            if (d.id === "RUS") {

                dx = 400;
                dy = 50;
                x = 650;
                y = 80;

            }

            var scale = Math.max(1, Math.min(8, 0.9 / Math.max((dx) / width, (dy) / height))),
                translate = [(width / 2) + offsetLeft - scale * x, (height / 2) + offsetTop - scale * y];
            svg.transition()
                .duration(500)
                .call(zoom.translate(translate).scale(scale).event);
        }

        function reset() {
            active.classed("active", false);
            active = d3.select(null);

            svg.transition()
                .duration(500)
                .call(zoom.translate([offsetLeft, offsetTop]).scale(1).event);
            // topFiveCountriesDiv.show();
            countryDetailsDiv.hide();
        }

        function zoomed() {
            g.style("stroke-width", 1.5 / d3.event.scale + "px");
            g.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
        }


        function stopped() {
            if (d3.event.defaultPrevented) d3.event.stopPropagation();
        }

    }

    $('.pswp').on('click', '.figure-download,.table-download', function () {
        window.open($(this).attr('href'), $(this).attr('target'));

        return false;
    });
    $('.figure-download,.table-download').on('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        window.open($(this).attr('href'), $(this).attr('target'));
        return false;
    });
    $('.indexData').on('check', function () {
        var visibleListElements = $(this).find('*.item-visible[data-filterdata]');

        if (visibleListElements.length == 0) {
            // console.log($(this).find('h3').text().trim() +' hide');


            $(this).hide();
        } else {
            //console.log($(this).find('h3').text().trim() +' show');
            $(this).show();
        }
    });

    window.blockMenuHeaderScroll = false;

    $(window).on('touchstart', function (e) {
        if ($(e.target).closest('.pswp__scroll-wrap').length == 1) {
            blockMenuHeaderScroll = true;
        }
    });

    $(window).on('touchend', function () {
        blockMenuHeaderScroll = false;
    });

    $(window).on('touchmove', function (e) {
        if (blockMenuHeaderScroll) {
            e.stopImmediatePropagation();
            e.stopPropagation();
            e.preventDefault();

            return false;
        }
    });
    //
    $('.pswp').on('click pswpTap mousedown mouseover  wheel mousewheel DOMMouseScroll', '.captionLong', function (e) {
        e.stopImmediatePropagation();
        e.stopPropagation();
        e.preventDefault();

        return false;
    });

    $('.close-icon').on('click', function () {
        $('.data-filter').trigger('reset');
    });
    $('.data-filter').on('reset', function () {
        $(this).val("");
        var closeIcon = $('.close-icon');
        var filterTargetSelector = $(this).attr('data-filtertarget');
        var filterTarget = $(filterTargetSelector);
        filterTarget.find('*[data-filterdata]').each(function () {
            $(this).addClass('item-visible').show();

        });
        closeIcon.hide();


        $('.indexData').trigger('check');

    })
        .on('keyup change', function () {
            var searchText = $(this).val();
            var closeIcon = $(this).next('.input-group-btn:hidden');
            var filterTargetSelector = $(this).attr('data-filtertarget');
            var filterTarget = $(filterTargetSelector);
            if (searchText.length > 0) {

                closeIcon.show();
            } else {
                $(this).trigger('reset');
                return true;
            }

            if (filterTarget.length == 0) {
                return false;
            }
            var showElements = filterTarget.find('*[data-filterdata*="' + searchText.toLowerCase() + '"]');
            var hideElements = filterTarget.find('*[data-filterdata]').not('*[data-filterdata*="' + searchText.toLowerCase() + '"]');
            showElements.each(function () {
                $(this).addClass('item-visible').show();

            });
            hideElements.each(function () {
                $(this).removeClass('item-visible').hide();
                $(this).next('ul').hide();
            });
            $('.indexData').trigger('check');

        });
    $('.data-filter').trigger('reset');
    $('.desktop-share').unbind().on('click', function (event) {
        event.preventDefault();
        event.stopPropagation();
        let element = $(this);
        let linkValue = $(element).data('href');

        updateClipboard(linkValue, element);

        return false;
    });

    /**
     *
     */
    $(document).on('click', '.search-pagination', function (event) {
        event.stopPropagation();
        event.stopImmediatePropagation();
        event.preventDefault();

        let element = $(this);
        let href = $(element).attr('href');
        let form = $('#ajaxSearch');

        if (href.length > 0) {
            href += '&ajax=true';
            $.ajax({
                method: 'GET',
                url: href,
                crossDomain: true,
                xhrFields: {
                    withCredentials: true
                },
                headers: {'X-Requested-With': 'XMLHttpRequest'}
            }).done(function (result) {
                loader.hide();
                lightbox.hide();
                form.trigger('ajax-form-done', [result, 'searchResult', form]);
            }).fail(function (result) {
                loader.hide();
                lightbox.hide();
                form.trigger('ajax-form-fail', [result, 'searchResult', form]);
            });

        }

        return false;

    });

    /**
     * Mantis#31008
     * @type {string}
     */
    let currentUrl = location.href;
    window.msTypePreselected = false;
    if (currentUrl.indexOf('by_ms_types') !== -1) {
        let iframe = $('#topics');
        let param = window.location.search.substr(1);
        if (iframe.length > 0) {
            let newIframeSrc = $(iframe).attr('src') + '&' + param;
            $(iframe).attr('src', newIframeSrc);
        }
    }
    if (currentUrl.indexOf('ms_types.php') !== -1) {
        let param = window.location.search.substr(1);
        let paramsArrayTmp = param.split("&");
        let paramSelection, paramTypeOfPaper;

        if (paramsArrayTmp.length > 0 && paramsArrayTmp.hasOwnProperty(1) && paramsArrayTmp[1].indexOf('ms-type-id') !== -1) {
            param = paramsArrayTmp[1];
        }

        if (paramsArrayTmp.length > 0 && paramsArrayTmp.hasOwnProperty(2) && paramsArrayTmp[2].indexOf('selection') !== -1) {
            paramSelection = paramsArrayTmp[2];
        }

        if (paramsArrayTmp.length > 0 && paramsArrayTmp.hasOwnProperty(3) && paramsArrayTmp[3].indexOf('preprints') !== -1) {
            paramTypeOfPaper = paramsArrayTmp[3];
        }

        if (param.length > 0 && param.indexOf('=') !== -1) {
            let paramsArray = param.split("=");
            if (paramsArray.hasOwnProperty(0) && paramsArray.hasOwnProperty(1) && paramsArray[0] === 'ms-type-id') {
                let msTypeId = paramsArray[1];
                let checkboxes = $('[name="manuscriptTypes[]"');
                if (checkboxes.length > 0) {
                    window.msTypePreselectedOnload = true;
                    for (let i = 0; i < checkboxes.length; i++) {
                        if (checkboxes.hasOwnProperty(i)) {
                            let checkbox = checkboxes[i];
                            if ($(checkbox).val() === msTypeId) {
                                let checkboxLabel = $(checkbox).closest('label').text();
                                let homeHeaderContent = $('.home-header').html();
                                $('.home-header').html(checkboxLabel);
                                $('.home-header').attr('data-prev-content', homeHeaderContent);
                                $(checkboxes[i]).prop('checked', true).trigger('change');
                                window.msTypePreselected = true;
                            } else {
                                $(checkboxes[i]).prop('checked', false).trigger('change');
                            }
                        }
                    }
                    window.msTypePreselectedOnload = false;
                }
            }
        }

        if (typeof paramSelection !== 'undefined' && paramSelection.length > 0 && paramSelection.indexOf('=') !== -1) {
            let paramsArray = paramSelection.split("=");
            if (paramsArray.hasOwnProperty(0) && paramsArray.hasOwnProperty(1) && paramsArray[0] === 'selection' && paramsArray[1] === 'none') {
                $('.manuscript-types-checkboxes-wrapper').removeClass('hide-on-fixed').hide();
                $('.ms-types-selection-wrapper').removeClass('show-on-fixed').show();
            }
        }

        window.addEventListener('check-paperListFilters-global', () => {
            if (typeof paramTypeOfPaper !== 'undefined' && paramTypeOfPaper.length > 0 && paramTypeOfPaper.indexOf('=') !== -1) {
                let paramsArray = paramTypeOfPaper.split("=");
                if (paramsArray.hasOwnProperty(0) && paramsArray.hasOwnProperty(1) && paramsArray[0] === 'preprints' && paramsArray[1] === 'none') {
                    $('input[name="paperListFilter"][value="final"]').prop('checked', true).trigger('click');
                    $('.radio').hide();
                }
            }
        });
    }

    function hideMathJax() {
        let styleElement = $('<style>');
        styleElement.html('.mjx-chtml,.math {display: none !important;}.disp-formula  svg{display: block !important;}.inline-formula svg{display: inline-block !important;}');
        styleElement.attr('id', 'hide-mathjax-styles');
        $('div.hspacePlaceholder').addClass('hspace').removeClass('hspacePlaceholder');
        $('.algorithmic-line-placeholder').addClass('algorithmic-line').removeClass('algorithmic-line-placeholder');
        $('body').append(styleElement);
    }

    function showMathJax() {
        $('div.hspace').addClass('hspacePlaceholder').removeClass('hspace');

        $('.algorithmic-line').each(function (index) {
            let hfillSpan = $(this).find('.hfill');
            if (hfillSpan.length == 0) {
                $(this).addClass('algorithmic-line-placeholder').removeClass('algorithmic-line');
            }
        });
        MathJax.Hub.Queue(["setRenderer", MathJax.Hub, "HTML-CSS"]);
        MathJax.Hub.Queue(["Rerender", MathJax.Hub]);
    }

    if ($('#mathjax-turn').length > 0) {
        hideMathJax();
    }
    window.mathjaxStatus = false;
    $('#mathjax-turn, .mathjax-turn').click(function (event) {
        if (window.mathjaxStatus === true) {
            hideMathJax();
            $(this).removeClass('btn-danger');
            $(this).addClass('btn-success');
            $(this).html('<i class="fal fa-function"></i> Turn MathJax on');
            window.mathjaxStatus = false;
        } else {
            $('#hide-mathjax-styles').remove();
            showMathJax();
            $(this).removeClass('btn-success');
            $(this).addClass('btn-danger');
            $(this).html('<i class="fal fa-function"></i> Turn MathJax off');
            window.mathjaxStatus = true;
        }
    });

    //0041146: Journal article and preprint HTML pages: remove Twitter share logo
    const twitterLink = $('a[title="Twitter"]');
    if(twitterLink.length > 0) {
        $(twitterLink).closest('div.col-auto').remove();
    }
});
/**
 * Method for copy something to clipboard
 * @param newClip
 * @param element
 */
window.updateClipboard = function (newClip, element) {
    navigator.clipboard.writeText(newClip).then(function () {

    }, function (err) {
        console.log('Error during copying to clipboard: ', err.toString());
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
            return console.log('Successful share');
        })
            .catch(function (error) {
                return console.log('Error sharing', error);
            });
    } else {
        let linkValue = $(element).data('href');
        updateClipboard(linkValue, element);
    }
    return false;
};
