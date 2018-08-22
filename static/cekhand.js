function cekHand(){
	$.ajax({
		type : 'POST',
		url : "/cekHand",
		contentType: 'application/json;charset=UTF-8',
		data:state="1",
		success: function(data) {
			//var divs = document.getElementsByClassName("gestBox1");
            //if(data=="True"){
			if(data['0']){
				$(".gestBox1").css({top:data['2']});
				$(".gestBox1").css({right:data['1']});
				$("#handin").css({position:"absolute"});
				$(".gestBox1").html('<img id=handBox src="../static/Pict/gestureOn.png" class="gestureIcon">');
			}else if(!data['0']){
				$(".gestBox1").css({top:0});
				$(".gestBox1").css({right:0});
				//$("#handin").removeClass("handin");
				$(".gestBox1").html('<img id=handBox src="../static/Pict/gestureOff.png" class="gestureIcon">');
				//alert("gesture:"+data);
			}
			//gesture();
			//statcekhand=setTimeout(function(){cekHand();},500); //delay is in milliseconds 
		}
	});
}
function gestonoff(stat){
    if(stat=="ok"){
        $("#inputcommand").html("Start Gesture");
    }else{
        $("#inputcommand").html("Move your right hand up");
    }
}
//statcekhand=setInterval(function(){cekHand();},500); //delay is in milliseconds 
