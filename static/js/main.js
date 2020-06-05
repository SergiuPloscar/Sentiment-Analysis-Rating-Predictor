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
});