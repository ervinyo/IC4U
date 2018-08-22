var setTimeoutFunction = null;

function homeButtonSelection(direction){
	var selectedIndex = -1;
	var buttons = $(".homeContent a");
	
	buttons.each(function(i){
		if($(this).hasClass("selected")){
			selectedIndex = i;
		}
	});
	
	if(selectedIndex == -1){
		buttons.eq(0).addClass("selected");
		return;
	}

	switch(direction){
		case "up":
			if(selectedIndex > 2){
				selectedIndex -= 3;
			}else{
				selectedIndex += 3;
			}
		break;
		case "down":
			if(selectedIndex < 3){
				selectedIndex += 3;
			}else{
				selectedIndex -= 3;
			}
		break;
		case "left":
			if(selectedIndex == 0){
				selectedIndex = buttons.length - 1;
			}else{
				selectedIndex--;
			}
		break;
		case "right":
			if(selectedIndex == buttons.length - 1){
				selectedIndex = 0;
			}else{
				selectedIndex++;
			}
		break;
	}
	buttons.removeClass("clicked");
	buttons.removeClass("selected");
	buttons.eq(selectedIndex).addClass("selected").focus();
	localStorage.setItem('clickedButtonIndex', selectedIndex);
}

function homeButtonClicked(){
	$(".homeContent a.selected").addClass("clicked");
	
	clearTimeout(setTimeoutFunction);
	setTimeoutFunction = setTimeout(function(){
		if($(".homeContent a.selected.clicked").length > 0){
			window.location = $(".homeContent a.selected.clicked").attr("href");
			localStorage.setItem('clickedButtonIndex', $(".homeContent a").index($(".homeContent a.selected.clicked")));
		}
	}, 1500);
}

$(document).on('keydown', function(e){
	switch(e.which) {
		case 13: // enter
			homeButtonClicked();
			return;
		case 37: // left
			homeButtonSelection("left");
		break;
		case 38: // up
			homeButtonSelection("up");
		break;
		case 39: // right
			homeButtonSelection("right");
		break;
		case 40: // down
			homeButtonSelection("down");
		break;
		default: return; // exit this handler for other keys
	}
	e.preventDefault();
});

$(".homeContent a").hover(function(){
	$(".homeContent a").removeClass("selected");
	$(".homeContent a").removeClass("clicked");
	$(this).addClass("selected");
},function(){
	$(".homeContent a").removeClass("selected");
	$(".homeContent a").removeClass("clicked");
});

$(".homeContent a").click(function(e){
	homeButtonClicked();
	e.preventDefault();
});

if(localStorage.getItem('clickedButtonIndex') != null){
	$(".homeContent a").removeClass("clicked");
	$(".homeContent a").removeClass("selected");
	$(".homeContent a").eq(localStorage.getItem('clickedButtonIndex')).addClass("selected").focus();
}else{
	localStorage.setItem('clickedButtonIndex', 0);
	$(".homeContent a").eq(0).addClass("selected").focus();
}