function show_javascript_only_elements()
{
	var elementsToMakeVisible = co_getElementsByClassName("JSjavascriptOnly");
//	for(index in elementsToMakeVisible)
	for(index=0, l=elementsToMakeVisible.length; index<l; index++)
	{
		if(typeof elementsToMakeVisible[index].style != "undefined") elementsToMakeVisible[index].style.visibility = "visible";
	}

	if(document.all && window.external)
	{
		var elementsToMakeVisible = co_getElementsByClassName("JSieOnly");

//		for(index in elementsToMakeVisible)
		for(index=0, l=elementsToMakeVisible.length; index<l; index++)
		{
			if(typeof elementsToMakeVisible[index].style != "undefined") elementsToMakeVisible[index].style.visibility = "visible";
		}
	}
}

function hotFixForACP(){

var oldPaper = document.getElementById("generator");

if(oldPaper){
	var jBanner = document.getElementById("j-banner");
	if(jBanner){
	jBanner.classList.remove("hide-on-tablet");
	jBanner.classList.remove("hide-on-mobile");
	}
		
	var jMobileBanner = document.getElementById("j-mobile-banner");
		if(jMobileBanner){
	jMobileBanner.style.display = 'none';
	}
	
	
	var mobileNavigationWrapper = document.getElementById("mobile-navigation-wrapper");
	if(mobileNavigationWrapper){
	mobileNavigationWrapper.style.display = 'none';
	}
	
		var mobileSearch = document.getElementById("mobile-search");
		if(mobileSearch){
		mobileSearch.style.display = 'none';
		}
	
	
	var jTopic = document.getElementById("j-topic");
	if(jTopic){
	jTopic.style.top = '1px';
	}
	
}

}

add_onload_action("show_javascript_only_elements()");
add_onload_action("hotFixForACP()");