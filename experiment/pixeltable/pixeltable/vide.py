import pixeltable as pxt
#print(dir(pxt))
#import pixeltable as pxt

import pixeltable as pxt

# Get raw DataFrame (no styling)
#df = pxt.list_functions()

# Filter for embedding-related functions
#filtered = df[df['name'].str.contains('embed|transformer|clip', case=False, na=False)]

# Print relevant columns clearly
#print(filtered[['name', 'return_type', 'args', 'description']].to_string(index=False))
frames = pxt.get_table("zeus_cache.demo_video_6dd4_table_frames")
print(frames.describe)


