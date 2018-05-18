
// TABS JQUERY
$("#tabs").tabs();


//FILE UPLOAD:
var file;
$('input[type=file]').on('change', prepareUpload);
function prepareUpload(event)
{
  console.log("Welcome"); 
  file = event.target.files;
}
$('form').on('submit', uploadFiles);
function uploadFiles(event)
{
    event.stopPropagation(); // Stop stuff happening
    event.preventDefault(); // Totally stop stuff happening

    // START A LOADING SPINNER HERE

    // Create a formdata object and add the files
   
    $.ajax({
        url: 'http://0.0.0.0:8080/upload',
        type: 'POST',
        data: file,
        dataType: 'json',
        cache: false,
        processData: false, // Don't process the files
        contentType: false, // Set content type to false as jQuery will tell the server its a query string request
        success: function(data, textStatus, jqXHR)
        {
            if(typeof data.error === 'undefined')
            {
                // Success so call function to process the form
                submitForm(event, data);
            }
            else
            {
                // Handle errors here
                console.log('ERRORS: ' + data.error);
            }
        },
        error: function(jqXHR, textStatus, errorThrown)
        {
            // Handle errors here
            console.log('ERRORS: ' + textStatus);
            // STOP LOADING SPINNER
        }
    });
}
