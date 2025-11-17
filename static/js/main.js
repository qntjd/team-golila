async function generate() {
  const genre = document.getElementById('genre').value;
  const bpm = document.getElementById('bpm').value;
  document.getElementById('status').innerText = 'Starting generation...';
  const res = await fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({genre, bpm})
  });
  const data = await res.json();
  document.getElementById('status').innerText = 'Generation started. Polling for result...';
  // Poll for latest file
  const poll = setInterval(async () => {
    const latest = await fetch('/latest');
    const j = await latest.json();
    if (j.file) {
      clearInterval(poll);
      document.getElementById('status').innerText = 'Generation complete: ' + j.file;
      if (j.file.endsWith('.mid')) {
        const link = '/outputs/' + j.file;
        document.getElementById('player').innerHTML = `<p>Download MIDI: <a href="${link}" download>${j.file}</a></p>`;
      } else if (j.file.endsWith('.wav')) {
        const link = '/outputs/' + j.file;
        document.getElementById('player').innerHTML = `<audio controls src="${link}"></audio><p><a href="${link}" download>Download</a></p>`;
      }
    }
  }, 2000);
}

document.getElementById('generateBtn').addEventListener('click', generate);