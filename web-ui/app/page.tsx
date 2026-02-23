"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LobbyPage() {
  const router = useRouter();
  const [name, setName] = useState("User");
  const [roomName, setRoomName] = useState("avatar-room");
  const [greeting, setGreeting] = useState("Hello! How can I help you today?");
  const [instructions, setInstructions] = useState(
    "You are a helpful conversational assistant. Be concise and friendly."
  );
  const [loading, setLoading] = useState(false);

  const handleJoin = async () => {
    setLoading(true);
    try {
      const metadata = { greeting, instructions };
      const res = await fetch("/api/token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          roomName,
          participantName: name,
          metadata,
        }),
      });

      if (!res.ok) throw new Error("Token fetch failed");

      const { token } = await res.json();

      // Encode params for the room page
      const params = new URLSearchParams({
        token,
        room: roomName,
      });
      router.push(`/room?${params.toString()}`);
    } catch (err) {
      console.error(err);
      alert("Failed to join room. Check console.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="lobby">
      <h1>🤖 AI Avatar</h1>
      <p>
        Start a voice & video call with your AI avatar. The avatar will see your
        camera, hear your mic, and respond with lip-synced speech.
      </p>

      <div className="lobby-card">
        <div className="field">
          <label>Your Name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter your name"
          />
        </div>

        <div className="field">
          <label>Room Name</label>
          <input
            value={roomName}
            onChange={(e) => setRoomName(e.target.value)}
            placeholder="avatar-room"
          />
        </div>

        <div className="field">
          <label>Greeting (what the avatar says first)</label>
          <input
            value={greeting}
            onChange={(e) => setGreeting(e.target.value)}
            placeholder="Hello! How can I help you today?"
          />
        </div>

        <div className="field">
          <label>System Instructions</label>
          <textarea
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            placeholder="You are a helpful assistant..."
          />
        </div>

        <button
          className="btn btn-primary"
          onClick={handleJoin}
          disabled={loading || !name.trim()}
        >
          {loading ? "Connecting…" : "🎙️ Start Call"}
        </button>
      </div>
    </div>
  );
}
