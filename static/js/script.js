



document.getElementById('send-btn').addEventListener('click', function() {
    var userInput = document.getElementById('user-input').value;
    if (userInput) {
        appendChatLog('User', userInput);
        document.getElementById('user-input').value = '';
        fetch('/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: userInput }),
        })
        .then(response => response.json())
        .then(data => appendChatLog('Model', data.response));
    }
});

document.getElementById('record-btn').addEventListener('click', function() {
    // Implement audio recording and sending to the server
    // This part requires additional JavaScript for recording audio
    // and then sending it to the '/process_audio' endpoint
});

function appendChatLog(sender, message) {
    var chatLogs = document.getElementById('chat-logs');
    var newLog = document.createElement('p');
    newLog.textContent = `${sender}: ${message}`;
    chatLogs.appendChild(newLog);
    chatLogs.scrollTop = chatLogs.scrollHeight;
}




