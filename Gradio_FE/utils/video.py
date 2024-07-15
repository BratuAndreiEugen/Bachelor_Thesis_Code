class VideoUtils:
    def __init__(self):
        pass

    @staticmethod
    def generate_video_html(video_path):
        html_template = f"""
            <video width="600" controls>
                <source src="{video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
        print(html_template)
        return html_template