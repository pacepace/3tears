/**
 * What plays (title, artist, album, artwork) and what play/pause/stop do. Handlers are installed
 * once; metadata is replaced as the item changes.
 *
 * For a live stream (`live: true`, the default) the seek and skip actions are cleared, so the
 * platform shows no scrubber or skip buttons, and no position is reported.
 *
 * Artwork: give absolute or same-origin URLs the platform can fetch by itself (iOS fetches them
 * outside the page), several sizes where there are any (`{ src, sizes: "512x512", type }`).
 */
export class MediaSessionControl {
    /**
     * @param {Navigator} [nav]
     * @param {{ onPlay?: () => unknown, onPause?: () => unknown, onStop?: () => unknown, live?: boolean,
     *           MediaMetadata?: typeof MediaMetadata }} [options]
     */
    constructor(nav?: Navigator, options?: {
        onPlay?: () => unknown;
        onPause?: () => unknown;
        onStop?: () => unknown;
        live?: boolean;
        MediaMetadata?: typeof MediaMetadata;
    });
    /** @type {MediaSession | null} */
    session: MediaSession | null;
    Metadata: {
        new (init?: MediaMetadataInit): MediaMetadata;
        prototype: MediaMetadata;
    };
    /** @type {string | null} */
    key: string | null;
    /** Whether the platform has a Media Session at all. */
    get supported(): boolean;
    /**
     * Installs (or, with null, clears) one action's handler; false where the platform lacks it. A
     * handler that throws, or returns a promise that rejects (`audio.play()` refused), is logged.
     * @param {string} action
     * @param {(() => unknown) | null | undefined} fn
     * @returns {boolean}
     */
    handle(action: string, fn: (() => unknown) | null | undefined): boolean;
    /**
     * What plays now. Unchanged metadata is not set again (setting it makes some platforms refetch
     * the artwork); metadata the platform refused is tried again next time.
     * @param {{ title: string, artist?: string, album?: string,
     *           artwork?: Array<{ src: string, sizes?: string, type?: string }> }} item
     */
    setNowPlaying({ title, artist, album, artwork }: {
        title: string;
        artist?: string;
        album?: string;
        artwork?: Array<{
            src: string;
            sizes?: string;
            type?: string;
        }>;
    }): void;
    /**
     * Tells the platform whether it plays (the lock screen's play/pause button follows it).
     * @param {boolean} playing
     */
    setPlaying(playing: boolean): void;
    /** Nothing plays any more: the lock screen forgets it. */
    clear(): void;
}
