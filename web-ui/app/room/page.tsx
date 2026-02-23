"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import {
  Room,
  RoomEvent,
  Track,
  RemoteTrack,
  RemoteTrackPublication,
  RemoteParticipant,
  ConnectionState,
} from "livekit-client";

function RoomInner() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const token = searchParams.get("token") || "";
  const roomName = searchParams.get("room") || "avatar-room";

  const livekitUrl = process.env.NEXT_PUBLIC_LIVEKIT_URL || "";

  const roomRef = useRef<Room | null>(null);
  const avatarVideoRef = useRef<HTMLVideoElement>(null);
  const avatarAudioRef = useRef<HTMLAudioElement>(null);

  const [connState, setConnState] = useState<string>("connecting");
  const [micEnabled, setMicEnabled] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [agentConnected, setAgentConnected] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [debugLog, setDebugLog] = useState<string[]>([]);

  const log = useCallback((msg: string) => {
    console.log("[Room]", msg);
    setDebugLog((prev) => [...prev.slice(-19), msg]);
  }, []);

  // Helper to attach remote tracks
  const attachTrack = useCallback(
    (track: RemoteTrack) => {
      if (track.kind === Track.Kind.Video && avatarVideoRef.current) {
        log(`Attaching video track`);
        track.attach(avatarVideoRef.current);
      }
      if (track.kind === Track.Kind.Audio && avatarAudioRef.current) {
        log(`Attaching audio track`);
        track.attach(avatarAudioRef.current);
        // Force play in case autoplay is blocked
        avatarAudioRef.current.play().catch(() => {});
      }
    },
    [log]
  );

  // ── Connect to room ──────────────────────────────────────
  useEffect(() => {
    if (!token || !livekitUrl) {
      router.push("/");
      return;
    }

    let cancelled = false;

    const connectRoom = async () => {
      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
      });
      roomRef.current = room;

      // Track subscribed — attach video/audio from the agent
      room.on(
        RoomEvent.TrackSubscribed,
        (
          track: RemoteTrack,
          _pub: RemoteTrackPublication,
          participant: RemoteParticipant
        ) => {
          log(
            `TrackSubscribed: ${track.kind} from ${participant.identity}`
          );
          attachTrack(track);
          if (track.kind === Track.Kind.Video) {
            setAgentConnected(true);
          }
        }
      );

      room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
        log(`TrackUnsubscribed: ${track.kind}`);
        track.detach();
      });

      // Agent/participant events
      room.on(RoomEvent.ParticipantConnected, (participant) => {
        log(`ParticipantConnected: ${participant.identity}`);
        setAgentConnected(true);
      });

      room.on(RoomEvent.ParticipantDisconnected, (participant) => {
        log(`ParticipantDisconnected: ${participant.identity}`);
        setAgentConnected(false);
      });

      room.on(
        RoomEvent.ConnectionStateChanged,
        (state: ConnectionState) => {
          log(`ConnectionState: ${state}`);
          setConnState(state);
        }
      );

      // Data messages (transcripts from livekit-agents)
      room.on(RoomEvent.DataReceived, (data: Uint8Array) => {
        try {
          const msg = JSON.parse(new TextDecoder().decode(data));
          if (msg.type === "transcript" || msg.text) {
            setTranscript(msg.text || msg.transcript || "");
            setTimeout(() => setTranscript(""), 5000);
          }
        } catch {
          // ignore non-JSON data
        }
      });

      room.on(RoomEvent.Disconnected, (reason) => {
        log(`Disconnected: ${reason ?? "unknown"}`);
        setConnState("disconnected");
      });

      // ── Connect ──
      try {
        log(`Connecting to ${livekitUrl} ...`);
        await room.connect(livekitUrl, token);
        if (cancelled) return;

        log(`Connected! Room: ${room.name}`);
        setConnState("connected");

        // Attach any existing remote tracks (agent may already be in the room)
        room.remoteParticipants.forEach((p) => {
          log(`Existing participant: ${p.identity}`);
          setAgentConnected(true);
          p.trackPublications.forEach((pub) => {
            if (pub.track && pub.isSubscribed) {
              attachTrack(pub.track as RemoteTrack);
            }
          });
        });

        // ── Enable microphone (with graceful fallback) ──
        try {
          await room.localParticipant.setMicrophoneEnabled(true);
          setMicEnabled(true);
          log("Microphone enabled");
        } catch (micErr: unknown) {
          const msg =
            micErr instanceof Error ? micErr.message : String(micErr);
          log(`Mic error: ${msg}`);
          setMicError(
            "Microphone permission denied. Click the mic button to retry."
          );
          setMicEnabled(false);
          // Don't disconnect — the user can still see/hear the avatar
        }
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        log(`Connection failed: ${msg}`);
        console.error("Connection failed:", err);
        setConnState("failed");
      }
    };

    connectRoom();

    return () => {
      cancelled = true;
      roomRef.current?.disconnect();
    };
  }, [token, livekitUrl, router, log, attachTrack]);

  // ── Mic toggle ──────────────────────────────────────────
  const toggleMic = useCallback(async () => {
    const room = roomRef.current;
    if (!room) return;
    const next = !micEnabled;
    try {
      await room.localParticipant.setMicrophoneEnabled(next);
      setMicEnabled(next);
      setMicError(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setMicError(`Mic error: ${msg}`);
    }
  }, [micEnabled]);

  // ── Hang up ──────────────────────────────────────────────
  const hangUp = useCallback(() => {
    roomRef.current?.disconnect();
    router.push("/");
  }, [router]);

  const isConnecting =
    connState === "connecting" || connState === "reconnecting";

  return (
    <div className="room-container">
      {/* Header */}
      <div className="room-header">
        <h2>AI Avatar Call</h2>
        <div className="room-status">
          {agentConnected ? (
            <>
              <span className="dot" />
              <span>
                Avatar connected &middot; Room: <b>{roomName}</b>
              </span>
            </>
          ) : (
            <span>
              Waiting for avatar agent&hellip; &middot; Room: <b>{roomName}</b>
            </span>
          )}
        </div>
      </div>

      {/* Mic permission warning */}
      {micError && (
        <div
          style={{
            background: "#3a2000",
            color: "#ffb347",
            padding: "8px 16px",
            textAlign: "center",
            fontSize: "0.85rem",
          }}
        >
          ⚠️ {micError}
        </div>
      )}

      {/* Main video area */}
      <div className="room-body">
        {isConnecting && (
          <div className="connecting-overlay">
            <div className="spinner" />
            <p>Connecting to room&hellip;</p>
          </div>
        )}

        {connState === "failed" && (
          <div className="connecting-overlay">
            <p style={{ color: "#ff6b6b", fontSize: "1.1rem" }}>
              Connection failed. Check your credentials.
            </p>
            <button
              className="btn btn-primary"
              onClick={() => router.push("/")}
            >
              Back to Lobby
            </button>
          </div>
        )}

        {connState === "disconnected" && (
          <div className="connecting-overlay">
            <p style={{ fontSize: "1.1rem" }}>Call ended.</p>
            <button
              className="btn btn-primary"
              onClick={() => router.push("/")}
            >
              Back to Lobby
            </button>
          </div>
        )}

        <div className="avatar-video-wrapper">
          <video
            ref={avatarVideoRef}
            autoPlay
            playsInline
            muted={false}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
              display: agentConnected ? "block" : "none",
            }}
          />
          {!agentConnected && connState === "connected" && (
            <div className="avatar-placeholder">
              <div className="spinner" />
              <p style={{ marginTop: 12 }}>
                Waiting for avatar agent to join&hellip;
              </p>
            </div>
          )}
          {!agentConnected && connState !== "connected" && (
            <div className="avatar-placeholder">🤖</div>
          )}

          {transcript && (
            <div className="transcript-overlay">{transcript}</div>
          )}
        </div>

        {/* Audio element for agent audio — hidden */}
        <audio ref={avatarAudioRef} autoPlay />
      </div>

      {/* Controls */}
      <div className="room-controls">
        <button
          className={`ctrl-btn ${micEnabled ? "active" : "muted"}`}
          onClick={toggleMic}
          title={micEnabled ? "Mute mic" : "Unmute mic"}
        >
          {micEnabled ? "🎙️" : "🔇"}
        </button>

        {agentConnected && (
          <div className="audio-indicator">
            <div className="bar" />
            <div className="bar" />
            <div className="bar" />
            <div className="bar" />
            <div className="bar" />
          </div>
        )}

        <button className="ctrl-btn hangup" onClick={hangUp} title="End call">
          📞
        </button>
      </div>

      {/* Debug log (small, bottom-right) */}
      <div
        style={{
          position: "fixed",
          bottom: 80,
          right: 8,
          maxWidth: 340,
          maxHeight: 200,
          overflow: "auto",
          fontSize: "0.65rem",
          color: "#888",
          background: "rgba(0,0,0,0.6)",
          padding: "6px 8px",
          borderRadius: 6,
          fontFamily: "monospace",
          pointerEvents: "none",
          zIndex: 999,
        }}
      >
        {debugLog.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>
    </div>
  );
}

export default function RoomPage() {
  return (
    <Suspense
      fallback={
        <div className="room-container">
          <div className="connecting-overlay">
            <div className="spinner" />
            <p>Loading&hellip;</p>
          </div>
        </div>
      }
    >
      <RoomInner />
    </Suspense>
  );
}
