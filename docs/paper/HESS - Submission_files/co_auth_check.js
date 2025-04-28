function check_for_valid_authentication_timer()
{
	try
	{
		if(window.frames[window.frames.length-1].src!="")
		{
			x=window.frames[window.frames.length-1].location.href; 

document.getElementById('co_auth_check_authiframecontainer').src="http://contentmanager1.copernicus-journals.net/webservices/get_auth_iframe_content.php?u="+encodeURI(window.location.href);
			document.getElementById('co_auth_check_authiframecontainer').style.display="block";
		}
	}
	catch(e)
	{
	}	
}

function check_for_valid_authentication()
{
	if(document.getElementById&&document.getElementById('co_auth_check_authiframecontainer'))
	{
		document.getElementsByTagName("body")[0].appendChild(window.co_auth_check_authiframe);

		window.setTimeout(check_for_valid_authentication_timer, 1555);
	}
}

window.co_auth_check_authiframe = document.createElement("iframe");
window.co_auth_check_authiframe.style.display="none";
window.co_auth_check_authiframe.setAttribute("src", "http://contentmanager1.copernicus-journals.net/webservices/authtest.php?d="+encodeURI(window.location.href.replace(/^http:\/\/([^\/]*).*$/g, "http://$1")));
window.co_auth_check_authiframe.setAttribute("id", "_co_authcalleriframe");

//add_onload_action("check_for_valid_authentication()");