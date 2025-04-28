function co_in_array(a, value)
{
    for(var i = 0, l = a.length; i < l; i++) if(a[i] == value) return true;
    return false;
}

function co_elementHasClass(element, className)
{
    if(element.className == undefined || element.className.match == undefined)
        return false;

    return element.className && ( element.className==className || element.className.match( new RegExp( "\\b"+className+"\\b" )) );
}

//
// expects same domain origin !!
//
function co_modify_css_class(classname,element,new_value)
{
    var cssrules_key;

    for (var styleIndex = 0, l1=document.styleSheets.length; styleIndex < l1; styleIndex++)
    {
        if (document.styleSheets[styleIndex]['rules'])
        {
            cssrules_key = 'rules';
        }
        else if (document.styleSheets[styleIndex]['cssRules'])
        {
            cssrules_key = 'cssRules';
        }
        else
        {
            return false;
        }

        for (var ruleIndex = 0, l2=document.styleSheets[styleIndex][cssrules_key].length; ruleIndex < l2; ruleIndex++)
        {
            if (document.styleSheets[styleIndex][cssrules_key][ruleIndex].selectorText == classname && document.styleSheets[styleIndex][cssrules_key][ruleIndex].style[element])
            {
                document.styleSheets[styleIndex][cssrules_key][ruleIndex].style[element] = new_value;
                return true;
            }
        }

        if(document.styleSheets[styleIndex].insertRule)
        {
            document.styleSheets[styleIndex].insertRule(classname+' { '+element+': '+new_value+'; }',document.styleSheets[styleIndex][cssrules_key].length);
        }
        else if (document.styleSheets[styleIndex].addRule)
        {
            document.styleSheets[styleIndex].addRule(classname,element+': '+new_value+';');
        }
        else
        {
            return false;
        }

        return true;
    }
}

function co_hide_tablerow(node)
{
    if(typeof node == "string")
    {
        if(!_get(node)) return false;
        node = _get(node);
    }
    else if (typeof node == "undefined")
    {
        return false
    }
    node.style.display="none";

    return true;
}

function co_show_tablerow(node)
{
    if(typeof node == "string")
    {
        if(!_get(node)) return false;
        node = _get(node);
    }
    else if (typeof node == "undefined")
    {
        return false
    }

    var isMSie = /*@cc_on!@*/false;
    if(isMSie)
    {
        node.style.display="block";
    }
    else
    {
        node.style.display="table-row";
    }

    return true;
}

function co_getElementsByClassName(className, parentElement)
{
    var elements = new Array ();
    var parent = ( typeof parentElement == 'object' ) ? parentElement : window.document;

    if(!parent)
    {
        return elements;
    }

    var children = parent.getElementsByTagName( "*" );

    for ( var i = 0; i < children.length; i++ )
    {
        if ( co_elementHasClass(children[i], className) )
        {
            elements[elements.length] = children[i];
        }
    }
    return elements;
}

function findPos(obj)
{
    var curleft = curtop = 0;
    if (obj.offsetParent) {
        curleft = obj.offsetLeft
        curtop = obj.offsetTop
        while (obj = obj.offsetParent) {
            curleft += obj.offsetLeft
            curtop += obj.offsetTop
        }
    }
    return [curleft,curtop];
}

function addEvent(obj, evType, fn)
{
    if(typeof fn == "undefined")
    {
        return false;
    }
    if (obj.addEventListener){
        obj.addEventListener(evType, fn, true);
        return true;
    } else if (obj.attachEvent){
        var r = obj.attachEvent("on"+evType, fn, true);
        return r;
    } else {
        return false;
    }
}

function removeEvent(obj, evType, fn)
{
    if(typeof fn == "undefined")
    {
        return false;
    }

    if (obj.removeEventListener){
        obj.removeEventListener(evType, fn, true);
        return true;
    } else if (obj.detachEvent){
        var r = obj.detachEvent("on"+evType, fn, true);
        return r;
    } else {
        return false;
    }
}

//
// getPageSize()
// Returns array with page width, height and window width, height
// Core code from - quirksmode.org
// Edit for Firefox by pHaez
// grabbed from http://www.huddletogether.com/projects/lightbox/lightbox.js
//
function getPageSize(){

    var xScroll, yScroll;

    if (window.innerHeight && window.scrollMaxY) {
        xScroll = document.body.scrollWidth;
        yScroll = window.innerHeight + window.scrollMaxY;
    } else if (document.body.scrollHeight > document.body.offsetHeight){ // all but Explorer Mac
        xScroll = document.body.scrollWidth;
        yScroll = document.body.scrollHeight;
    } else { // Explorer Mac...would also work in Explorer 6 Strict, Mozilla and Safari
        xScroll = document.body.offsetWidth;
        yScroll = document.body.offsetHeight;
    }

    var windowWidth, windowHeight;
    if (self.innerHeight) {	// all except Explorer
        windowWidth = self.innerWidth;
        windowHeight = self.innerHeight;
    } else if (document.documentElement && document.documentElement.clientHeight) { // Explorer 6 Strict Mode
        windowWidth = document.documentElement.clientWidth;
        windowHeight = document.documentElement.clientHeight;
    } else if (document.body) { // other Explorers
        windowWidth = document.body.clientWidth;
        windowHeight = document.body.clientHeight;
    }

    // for small pages with total height less then height of the viewport
    if(yScroll < windowHeight){
        pageHeight = windowHeight;
    } else {
        pageHeight = yScroll;
    }

    // for small pages with total width less then width of the viewport
    if(xScroll < windowWidth){
        pageWidth = windowWidth;
    } else {
        pageWidth = xScroll;
    }


    arrayPageSize = new Array(pageWidth,pageHeight,windowWidth,windowHeight)
    return arrayPageSize;
}
//
// co_getPageScroll()
// Returns array with x,y page scroll values.
// Core code from - quirksmode.org
// grabbed from http://www.huddletogether.com/projects/lightbox/lightbox.js
//
function co_getPageScroll(){

    var yScroll;

    if (self.pageYOffset) {
        yScroll = self.pageYOffset;
    } else if (document.documentElement && document.documentElement.scrollTop){	 // Explorer 6 Strict
        yScroll = document.documentElement.scrollTop;
    } else if (document.body) {// all other Explorers
        yScroll = document.body.scrollTop;
    }

    arrayPageScroll = new Array('',yScroll)
    return arrayPageScroll;
}

function co_get_viewport_size()
{
    var viewportwidth;
    var viewportheight;

    // the more standards compliant browsers (mozilla/netscape/opera/IE7) use window.innerWidth and window.innerHeight

    if (typeof window.innerWidth != 'undefined')
    {
        viewportwidth = window.innerWidth,
            viewportheight = window.innerHeight
    }

// IE6 in standards compliant mode (i.e. with a valid doctype as the first line in the document)

    else if (typeof document.documentElement != 'undefined'
        && typeof document.documentElement.clientWidth !=
        'undefined' && document.documentElement.clientWidth != 0)
    {
        viewportwidth = document.documentElement.clientWidth,
            viewportheight = document.documentElement.clientHeight
    }

    // older versions of IE

    else
    {
        viewportwidth = document.getElementsByTagName('body')[0].clientWidth,
            viewportheight = document.getElementsByTagName('body')[0].clientHeight
    }

    viewportSize = new Array(viewportwidth,viewportheight  )
    return viewportSize ;

}

function getStyle(el,styleProp)
{
    if(typeof el != "object" )
    {
        if(document.getElementById(el))
        {
            var obj = document.getElementById(el);
        }
        else
        {
            return false;
        }
    }
    else
    {
        var obj = el;
    }

    var ret = false;


    // Handle "float" property as a special case
    if (styleProp=="float")
    {
        ret = getStyle(obj,"cssFloat");
        if (ret==null)
        {
            ret = getStyle(obj,"styleFloat");
        }

        return ret;
    }

    if(obj.style && obj.style[toCamelCase(styleProp)])
    {
        ret = obj.style[toCamelCase(styleProp)];
    }
    else if (obj.currentStyle && obj.currentStyle[toCamelCase(styleProp)])
        ret = obj.currentStyle[toCamelCase(styleProp)];
    else if (window.getComputedStyle)
        ret = document.defaultView.getComputedStyle(obj,null).getPropertyValue(styleProp);

    return ret;
}

/** toCamelCase(input)
 * Converts string input to a camel cased version of itself.
 * For example:
 * toCamelCase("z-index"); // returns zIndex
 * toCamelCase("border-bottom-style"); // returns borderBottomStyle.
 */
function toCamelCase(s) {
    for(var exp = toCamelCase.exp;
        exp.test(s); s = s.replace(exp, RegExp.$1.toUpperCase()) );
    return s;
}
toCamelCase.exp = /-([a-z])/;






window.onload_actions = new Array();

function add_onload_action(evalstring)
{
    window.onload_actions.push(evalstring);
}

function onload_action()
{
    for(i=0; i<window.onload_actions.length; i++)
    {
        try
        {
            eval(window.onload_actions[i]);
        }
        catch(e)
        {
            alert('onload action "'+window.onload_actions[i]+'" failed:'+e);
        }
    }
}


function _get(node)
{
    return document.getElementById(node);
}

function alertForDesktopRedirection() {
    var viewPortWidth = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
    if (viewPortWidth < 1025) {
        var message = document.querySelector('body').getAttribute('data-non-mobile-optimized-message');
        var nonMobileOptimizedElements = document.querySelectorAll('[data-non-mobile-optimized]');

        if (!message) {
            message = 'You are leaving the mobile-optimized environment. To proceed, please use an appropriate device.';
        }

        if (nonMobileOptimizedElements && nonMobileOptimizedElements.length > 0) {
            for (var i = 0; i < nonMobileOptimizedElements.length; i++) {
                nonMobileOptimizedElements[i].addEventListener("click", function (e) {
                    console.log("Calling non-mobile-message");
                    if (window.confirm(message)) {
                        return true;
                    } else {
                        e.preventDefault();
                        return false;
                    }
                });
            }
        }
    }
}


function detectAnchor(){
    var anchorValue = window.location.hash.substr(1);
    var anchorElement = document.getElementById(anchorValue);

    if(anchorElement){
//could be that we must use display: block instead
        anchorElement.style.display = 'inline-block';
        anchorElement.style.visibility = 'visible';
        if(anchorElement.offsetParent === null){
            showInvisibleParent(anchorElement);
        }
        anchorElement.scrollIntoView({block: "center", behavior: "smooth"});
    }

    attachEventListenerToAncors();
}

function attachEventListenerToAncors() {
    var links = document.querySelectorAll('a');
    if(links.length > 0){
        for(var i =0; i < links.length; i++){
            var link = links[i];

            link.addEventListener("click", function (e) {
                let element = e.currentTarget;
                let href = element.href;

                if(/^#/.test(href) || href.indexOf(window.location.href) > -1){
                    var anchorValue = href.substr(href.indexOf('#') + 1);
                    var anchoreElement = document.getElementById(anchorValue);
                    showInvisibleParent(anchoreElement);
                }
            });
        }
    }
}

function showInvisibleParent(element){
    if(element !== null){
        var parentElement = element.parentElement;

        if(parentElement && parentElement.offsetParent === null){
            //could be that we must use display: block instead
            parentElement.style.display = 'inline-block';
            parentElement.style.visibility = 'visible';
            if(parentElement.offsetParent === null){
                showInvisibleParent(parentElement);
            }
        }
    }
}

add_onload_action("alertForDesktopRedirection()");
add_onload_action("detectAnchor()");
addEvent(window, 'load', onload_action);
