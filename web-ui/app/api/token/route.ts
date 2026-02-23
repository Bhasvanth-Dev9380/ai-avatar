import { NextRequest, NextResponse } from "next/server";
import {
  AccessToken,
  RoomServiceClient,
  AgentDispatchClient,
} from "livekit-server-sdk";

export async function POST(req: NextRequest) {
  try {
    const { roomName, participantName, metadata } = await req.json();

    const apiKey = process.env.LIVEKIT_API_KEY;
    const apiSecret = process.env.LIVEKIT_API_SECRET;
    const livekitUrl = process.env.LIVEKIT_URL || "";

    if (!apiKey || !apiSecret) {
      return NextResponse.json(
        { error: "LiveKit credentials not configured" },
        { status: 500 }
      );
    }

    const room = roomName || "avatar-room";

    const token = new AccessToken(apiKey, apiSecret, {
      identity: participantName || "user",
      name: participantName || "User",
      metadata: metadata ? JSON.stringify(metadata) : undefined,
    });

    token.addGrant({
      room,
      roomJoin: true,
      roomCreate: true,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    });

    const jwt = await token.toJwt();

    // Ensure the room exists and dispatch our avatar agent
    try {
      const roomService = new RoomServiceClient(livekitUrl, apiKey, apiSecret);
      await roomService.createRoom({ name: room, emptyTimeout: 300 });

      const dispatchClient = new AgentDispatchClient(
        livekitUrl,
        apiKey,
        apiSecret
      );
      await dispatchClient.createDispatch(room, "avatar-agent", {
        metadata: metadata ? JSON.stringify(metadata) : undefined,
      });
      console.log(`Agent dispatched to room: ${room}`);
    } catch (dispatchErr) {
      // Log but don't fail — the user can still join, agent may auto-dispatch
      console.warn("Agent dispatch warning:", dispatchErr);
    }

    return NextResponse.json({ token: jwt });
  } catch (err) {
    console.error("Token generation error:", err);
    return NextResponse.json(
      { error: "Failed to generate token" },
      { status: 500 }
    );
  }
}
