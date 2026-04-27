export default async (request) => {
  const { text } = await request.json();
  // Set 'OPENROUTER_API_KEY' in your Netlify Environment Variables
  const key = Deno.env.get("OPENROUTER_API_KEY");

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: { "Authorization": `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      "model": "google/gemma-2-9b-it:free",
      "messages": [{ "role": "user", "content": `Turn these ASL letters into a natural sentence: ${text}` }]
    })
  });

  const data = await response.json();
  return new Response(JSON.stringify({ sentence: data.choices[0].message.content }));
};