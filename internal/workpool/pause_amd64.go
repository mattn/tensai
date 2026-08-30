package workpool

// pause executes the PAUSE instruction: it stalls the pipeline for tens
// of cycles, releasing execution resources to the sibling hyperthread,
// without yielding the P the way Gosched would.
func pause()
