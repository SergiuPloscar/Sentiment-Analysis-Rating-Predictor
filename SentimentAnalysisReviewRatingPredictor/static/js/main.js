$(document).ready(function(){
    $('#form').on('submit', function(event) {
        event.preventDefault();
        $.ajax({
        type: 'POST',
        url:  '/classify_review',
        data: $('#review-area').serialize(),
        success: function(data){

         if(data.error){
            $('#error').text(data.error).show();
            $('#rating').text("");
            $('#star').text("");
        }
        else{
            $('#error').hide();
            $('#rating').text(data.rating);
            $('#star').text("\u2606");
        }
        }
        })
    });
    $('#ex1b').click(function(){
        console.log("AIDEPLMMMM");
        var review = $("#ex1").text();
        console.log(review);
        $("#review-area").val(review);
        $("#submit").click();
    });
    $('#ex2b').click(function(){
        var review = $("#ex2").text();
        $("#review-area").val(review);
        $("#submit").click();
    });
    $('#ex3b').click(function(){
        var review = $("#ex3").text();
        $("#review-area").val(review);
        $("#submit").click();
    });
    $('#ex4b').click(function(){
        var review = $("#ex4").text();
        $("#review-area").val(review);
        $("#submit").click();
    });
    $('#ex5b').click(function(){
        var review = $("#ex5").text();
        $("#review-area").val(review);
        $("#submit").click();
    });
});