
// TABS JQUERY
$("#tabs").tabs().css({
    'min-height': '800px',
    'overflow': 'auto'
 });

$(".settings").hide();
$(".training").hide();
//FILE UPLOAD:
$('#upload').submit(function(event){
    event.stopPropagation();
    event.preventDefault();

    var file =  document.getElementById('dataFile').files[0];
    var reader = new FileReader();
    reader.readAsText(file, 'UTF-8');
    reader.onload = upload;

    function upload(event){
        var result =  event.target.result;
        var filename = file.name;
        $.post("http://0.0.0.0:8080/upload", {data: result, name: filename},function(data){
            $("#status").text(data);
            $("#status").css('background-color','#99ff99');
            setTimeout(function() {$("#status").css('background-color','').text("");}, 3000);
            display();
        });
      
    }
    function display(event){
        $(".settings").show();
        $.get("http://0.0.0.0:8080/csvhanlder", function(data){
               $(".displayData").empty();
               $(".displayData").append(data);
            });
    }
})

//Neuron Configuration 
$("#neuron").submit(function (event){
    event.preventDefault();
    event.stopPropagation();
    form_data = $("#neuron").serialize()
    $(".training").show();
    var viewport = $("#process");
    $.ajax({
        url: "http://0.0.0.0:8080/",
        method: "POST",
        data: form_data,
        success: function(response){
            var jsonResult = $.parseJSON(response);
            $.each(jsonResult.Process, function(index, value){
                $(".results").find('tbody')
                .append("<tr><td>"+value.Epoch+"</td><td>"+value.Accuracy+"% </td><td>"+value.Loss+"</td></tr>")
            })
            console.log(jsonResult)
            $(".overall_accuracy").text("Overall Accuracy: "+jsonResult.Acc);
            $(".train_error").text("Training Error: "+jsonResult.TrainE);
            $(".test_error").text("Test Error: "+jsonResult.TestE); 
        }
    })       
   
})