<script lang="ts">
  import { onMount, tick, untrack } from "svelte";
  import { get } from "svelte/store";
  import { createPerson, deletePerson, listPersonSummaries, mergePeople, renamePerson } from "../lib/api";
  import { includeNSFW, personEpoch, refreshPeople, setPerson } from "../lib/stores";
  import { canMergePeople, personFacesLabel } from "../lib/personPresentation";
  import type { Person, PersonSummary } from "../lib/types";
  import CollectionCovers from "./CollectionCovers.svelte";
  import ConfirmDialog from "./ConfirmDialog.svelte";

  let people = $state<PersonSummary[]>([]);
  let loading = $state(true);
  let pending = $state(false);
  let error = $state("");
  let newName = $state("");
  let renaming = $state<number | null>(null);
  let renameValue = $state("");
  let deleting = $state<Person | null>(null);
  let managing = $state<number | null>(null);
  let merging = $state<number | null>(null);
  let mergeTarget = $state("");
  let dragSource = $state<number | null>(null);
  let dropTarget = $state<number | null>(null);
  let mergeConfirm = $state<{ from: PersonSummary; into: PersonSummary } | null>(null);
  let active = true;
  let request = 0;
  let controller: AbortController | undefined;
  let loadedScope = $state("");
  const scope = $derived(`${$personEpoch}:${$includeNSFW}`);
  const visiblePeople = $derived(loadedScope === scope ? people : []);

  function focusManage(id: number) {
    document.querySelector<HTMLButtonElement>(`[data-person-manage="${id}"]`)?.focus();
  }
  function closeMenu() {
    if (managing !== null) focusManage(managing);
    managing = null;
  }
  function onKey(event: KeyboardEvent) {
    if (event.key === "Escape" && managing !== null) {
      event.preventDefault();
      closeMenu();
    }
  }
  function onClick(event: MouseEvent) {
    if (!(event.target instanceof Element) || !event.target.closest("[data-person-management]")) managing = null;
  }
  async function load() {
    const token = ++request;
    controller?.abort();
    const ac = new AbortController();
    controller = ac;
    const epoch = get(personEpoch);
    const key = `${epoch}:${get(includeNSFW)}`;
    const current = () => active && token === request && !ac.signal.aborted && epoch === get(personEpoch);
    loading = true;
    error = "";
    people = [];
    managing = null;
    try {
      const response = await listPersonSummaries({ signal: ac.signal });
      if (current()) {
        people = response.people;
        loadedScope = key;
      }
    } catch (err) {
      if (current()) error = err instanceof Error ? err.message : "Could not load people";
    } finally {
      if (current()) loading = false;
    }
  }
  $effect(() => {
    void scope;
    untrack(() => void load());
  });
  onMount(() => () => { active = false; controller?.abort(); });

  async function create(event: SubmitEvent) {
    event.preventDefault();
    if (pending) return;
    const trimmed = newName.trim();
    if (trimmed && Array.from(trimmed).length > 100) {
      error = "Use a person name of 1–100 characters.";
      return;
    }
    pending = true;
    error = "";
    try {
      await createPerson(trimmed || undefined);
      if (active) newName = "";
      refreshPeople();
    } catch (err) {
      if (active) error = err instanceof Error ? err.message : "Could not create person";
    } finally {
      if (active) pending = false;
    }
  }
  function startRename(person: PersonSummary) {
    closeMenu();
    renaming = person.id;
    renameValue = person.name;
    void tick().then(() => document.querySelector<HTMLInputElement>('[aria-label="New person name"]')?.focus());
  }
  async function saveRename(id: number) {
    const trimmed = renameValue.trim();
    if (!trimmed || Array.from(trimmed).length > 100) {
      error = "Use a person name of 1–100 characters.";
      return;
    }
    pending = true;
    error = "";
    try {
      await renamePerson(id, trimmed);
      if (active) { renaming = null; focusManage(id); }
      refreshPeople();
    } catch (err) {
      if (active) error = err instanceof Error ? err.message : "Could not rename person";
    } finally {
      if (active) pending = false;
    }
  }
  async function remove() {
    const item = deleting;
    deleting = null;
    if (!item || pending) return;
    pending = true;
    error = "";
    try {
      await deletePerson(item.id);
      refreshPeople();
    } catch (err) {
      if (active) error = err instanceof Error ? err.message : "Could not delete person";
    } finally {
      if (active) pending = false;
    }
  }
  function startMerge(id: number) {
    closeMenu();
    merging = id;
    mergeTarget = "";
  }
  async function confirmMerge(id: number) {
    const target = Number(mergeTarget);
    if (!canMergePeople(id, target)) {
      error = "Choose another person to merge into.";
      return;
    }
    await doMerge(id, target);
    if (active) merging = null;
  }
  async function doMerge(fromId: number, intoId: number) {
    pending = true;
    error = "";
    try {
      await mergePeople([fromId], intoId);
      if (active) refreshPeople();
    } catch (err) {
      if (active) error = err instanceof Error ? err.message : "Could not merge people";
    } finally {
      if (active) pending = false;
    }
  }
  function onCardDragStart(id: number, event: DragEvent) {
    if (pending || !event.dataTransfer) return;
    dragSource = id;
    dropTarget = null;
    event.dataTransfer.setData("text/plain", String(id));
    event.dataTransfer.effectAllowed = "move";
  }
  function onCardDragOver(id: number, event: DragEvent) {
    if (dragSource === null || !canMergePeople(dragSource, id) || !event.dataTransfer) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    dropTarget = id;
  }
  function onCardDragLeave(id: number, event: DragEvent) {
    const to = event.relatedTarget;
    if (to instanceof Node && event.currentTarget instanceof Node && event.currentTarget.contains(to)) return;
    if (dropTarget === id) dropTarget = null;
  }
  function onCardDrop(id: number, event: DragEvent) {
    event.preventDefault();
    const from = dragSource;
    dragSource = null;
    dropTarget = null;
    if (from === null || !canMergePeople(from, id)) return;
    const source = visiblePeople.find((p) => p.id === from);
    const target = visiblePeople.find((p) => p.id === id);
    if (source && target) mergeConfirm = { from: source, into: target };
  }
  function onCardDragEnd() {
    dragSource = null;
    dropTarget = null;
  }
  const button = "px-3 py-2 border border-line rounded-lg cursor-pointer disabled:opacity-50";
</script>

<svelte:window onkeydown={onKey} onclick={onClick} />

<section data-people-pane class="px-5 sm:px-9 py-5 max-w-5xl w-full">
  <p class="text-muted mb-4">Faces grouped automatically across your library. Rename people, merge duplicates, or open one to browse every photo and video they appear in. Tip: drag one person onto another to merge them.</p>
  <form onsubmit={create} class="flex flex-wrap items-center gap-2">
    <input aria-label="Person name" placeholder="Person name (blank for automatic)" bind:value={newName} disabled={pending} class="min-w-0 flex-1 px-3 py-2 bg-surface border border-line-2 rounded-lg" />
    <button type="submit" disabled={pending} class="px-3 py-2 bg-ink text-white rounded-lg cursor-pointer disabled:opacity-50">{pending ? "Saving…" : "Add person"}</button>
  </form>
  {#if loading}<p role="status" class="mt-4">Loading people…</p>{/if}
  {#if error}<p role="alert" class="text-bad mt-4">{error}</p><button class={button} onclick={load} disabled={pending}>Retry</button>{/if}
  {#if !loading && !error && visiblePeople.length === 0}<p class="mt-4 text-muted">No people yet. Face grouping runs in the background after indexing.</p>{/if}
  <ul class="list-none p-0 mt-5 grid md:grid-cols-2 gap-3">
    {#each visiblePeople as person (person.id)}
      <li
        data-person-card={person.id}
        data-person-drop={dropTarget === person.id ? "true" : undefined}
        draggable={!pending && renaming === null && merging === null}
        ondragstart={(event) => onCardDragStart(person.id, event)}
        ondragover={(event) => onCardDragOver(person.id, event)}
        ondragleave={(event) => onCardDragLeave(person.id, event)}
        ondrop={(event) => onCardDrop(person.id, event)}
        ondragend={onCardDragEnd}
        class="relative min-w-0 bg-surface border rounded-card flex items-stretch {dropTarget === person.id ? 'border-accent shadow-elev outline outline-2 outline-accent/60' : 'border-line'}"
      >
        {#if renaming === person.id}
          <div class="p-4 min-w-0 w-full">
            <form onsubmit={(e) => { e.preventDefault(); void saveRename(person.id); }} class="flex flex-wrap items-center gap-2">
              <input aria-label="New person name" bind:value={renameValue} disabled={pending} class="min-w-0 flex-1 px-3 py-2 bg-surface border border-line-2 rounded-lg" />
              <button type="submit" disabled={pending} class="px-3 py-2 bg-ink text-white rounded-lg cursor-pointer disabled:opacity-50">Save name</button>
            </form>
            <button class={`${button} mt-2`} onclick={() => { renaming = null; focusManage(person.id); }}>Cancel rename</button>
          </div>
        {:else if merging === person.id}
          <div class="p-4 min-w-0 w-full">
            <form onsubmit={(e) => { e.preventDefault(); void confirmMerge(person.id); }} class="flex flex-wrap items-center gap-2">
              <label class="w-full text-sm text-muted" for={`merge-target-${person.id}`}>Merge {person.name} into…</label>
              <select id={`merge-target-${person.id}`} bind:value={mergeTarget} disabled={pending} class="min-w-0 flex-1 px-3 py-2 bg-surface border border-line-2 rounded-lg">
                <option value="">Choose a person</option>
                {#each visiblePeople.filter((p) => p.id !== person.id) as target (target.id)}
                  <option value={target.id}>{target.name}</option>
                {/each}
              </select>
              <button type="submit" disabled={pending} class="px-3 py-2 bg-ink text-white rounded-lg cursor-pointer disabled:opacity-50">Merge</button>
            </form>
            <button class={`${button} mt-2`} onclick={() => { merging = null; focusManage(person.id); }}>Cancel merge</button>
          </div>
        {:else}
          <button class="flex-1 min-w-0 flex items-center gap-3 p-4 text-left rounded-card cursor-pointer hover:bg-surface-2 focus-visible:outline-accent" aria-label={`Browse ${person.name}`} onclick={() => setPerson(person.id)}>
            <CollectionCovers covers={person.covers} />
            <span class="min-w-0">
              <span class="block font-medium break-words [overflow-wrap:anywhere]">{person.name}</span>
              <span class="block text-sm text-muted mt-1">{personFacesLabel(person.face_count)}</span>
            </span>
          </button>
          <div data-person-management class="shrink-0 p-2 pl-0">
            <button data-person-manage={person.id} class="w-9 h-9 rounded-lg cursor-pointer hover:bg-surface-2 disabled:opacity-50" disabled={pending} aria-label={`Manage ${person.name}`} aria-expanded={managing === person.id} aria-controls={`person-manage-${person.id}`} onclick={() => managing = managing === person.id ? null : person.id}>⋯</button>
            {#if managing === person.id}
              <div id={`person-manage-${person.id}`} class="absolute right-2 top-12 z-10 grid gap-1 p-2 bg-surface border border-line rounded-lg shadow-elev">
                <button class={button} aria-label={`Rename ${person.name}`} onclick={() => startRename(person)}>Rename</button>
                <button class={button} aria-label={`Merge ${person.name} into another person`} onclick={() => startMerge(person.id)}>Merge into…</button>
                <button class={`${button} text-bad`} aria-label={`Delete ${person.name}`} onclick={() => { closeMenu(); deleting = person; }}>Delete person</button>
              </div>
            {/if}
          </div>
        {/if}
      </li>
    {/each}
  </ul>
</section>
{#if deleting}
  <ConfirmDialog title="Delete person?" detail={deleting.name} note="Only the grouping will be removed. Faces stay unassigned and original media stays in your library." confirmLabel="Delete person" onconfirm={() => void remove()} oncancel={() => deleting = null} />
{/if}
{#if mergeConfirm}
  <ConfirmDialog
    title="Merge people?"
    detail={`${mergeConfirm.from.name} → ${mergeConfirm.into.name}`}
    note={`All faces move to ${mergeConfirm.into.name}. ${mergeConfirm.from.name} will be deleted. This cannot be undone.`}
    confirmLabel="Merge"
    onconfirm={() => { const pair = mergeConfirm; mergeConfirm = null; if (pair) void doMerge(pair.from.id, pair.into.id); }}
    oncancel={() => mergeConfirm = null}
  />
{/if}
